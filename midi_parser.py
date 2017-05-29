import midi
import numpy as np
import glob

import music21

lowerBound = 24
upperBound = 102
span = upperBound-lowerBound

num_timesteps = 1

majors = dict([("A-", 4),("G#", 4),("A", 3),("A#", 2),("B-", 2),("B", 1),
               ("C", 0),("C#", -1),("D-", -1),
               ("D", -2),("D#", -3),("E-", -3),("E", -4),
               ("F", -5),("F#", 6),("G-", 6),("G", 5)])

minors = dict([("G#", 1), ("A-", 1),("A", 0),("A#", -1),("B-", -1),
               ("B", -2),("C", -3),("C#", -4),
               ("D-", -4),("D", -5),("D#", 6),("E-", 6),
               ("E", 5),("F", 4),("F#", 3),("G-", 3),("G", 2)])

def preprocess(path):
    for file in glob.glob("{}/*.mid*".format(path)):
        score = music21.converter.parse(file)
        key = score.analyze('key')
        
        if key.mode == "major":
            halfSteps = majors[key.tonic.name]
            
        elif key.mode == "minor":
            halfSteps = minors[key.tonic.name]
            
        newscore = score.transpose(halfSteps)
        key = newscore.analyze('key')
        #print key.tonic.name, key.mode
        newFileName = "{}/C_{}".format(path, file.split('/')[-1])
        newscore.write('midi', newFileName)
        

def get_song(song):
    # Reshape song vector by placing consecutive timesteps next to eachother
    song = np.array(song)
    # Round down to nearest multiple
    song = song[:int(np.floor((song.shape[0]/num_timesteps) * num_timesteps))]
    # Reshape into blocks of num_timesteps
    song = np.reshape(song, [song.shape[0]/num_timesteps, song.shape[1]*num_timesteps])
    return song
    

def get_songs(path):
    # Get all song vectors from a folder of MIDI files
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in files:
        try:
            song = np.array(midiToStatematrix(f))
            song = get_song(song)
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except:
            # Just ignore songs that can't be parsed
            continue         
    return songs


def write_song(path, song):
    #Reshape song into statematrix
    song = np.reshape(song, (song.shape[0]*num_timesteps, 2*span))
    song = clean_song(song)

    statematrixToMidi(song, name=path)
    

def clean_song(song, max_simul_notes=6):
    '''
    Takes a song statematrix with sparsely spaced chords that contain too
    many simultaneous chords and removes them. Can set the number of notes
    that are exceptable to be played at once with max_simul_notes arg.
    '''
    clean_song = []
    prevstate = [0 for i in range(span * 2)]
    
    for time, state in enumerate(song):
        if np.sum(state[:span]) > max_simul_notes:
            state = np.concatenate((prevstate[:span],[0] * span))
        elif np.sum(state[:span]) == 0:
            if np.random.uniform(0, 1, 1) < 0.9:
                state = np.concatenate((prevstate[:span],[0] * span))
        prevstate = state 
        clean_song.append(state)
        
    clean_song = np.array(clean_song)
    return clean_song


def get_melody_and_drums(midifile):
    pattern = midi.read_midifile(midifile)
    
    melody_pattern = midi.Pattern()
    drum_pattern = midi.Pattern()

    melody_pattern.resolution = pattern.resolution
    drum_pattern.resolution = pattern.resolution

    for track in pattern:
        total_ticks = [0] * 17
    
        melody_track = midi.Track()
        drum_track = midi.Track()
    
        melody_channels = [0]
        for i in xrange(len(track)):
            evt = track[i]
         
            if isinstance(evt, midi.ProgramChangeEvent):
                if evt.data in [[0], [1], [2], [4], [5]]:
                    melody_channels.append(evt.channel)
                    melody_track.append(evt)
                
            elif isinstance(evt, midi.TimeSignatureEvent) or \
                    isinstance(evt, midi.SetTempoEvent) or \
                    isinstance(evt, midi.KeySignatureEvent):          
                melody_track.append(evt)
                drum_track.append(evt)
            
            elif isinstance(evt, midi.NoteEvent):
                total_ticks = [x + evt.tick for x in total_ticks]
            
                if (evt.channel == 9):
                    if type(evt) == midi.events.NoteOnEvent:
                        add_event = midi.NoteOnEvent()
                    else:
                        add_event = midi.NoteOffEvent()

                    add_event.tick = total_ticks[9]
                    add_event.channel = evt.channel
                    add_event.data = evt.data

                    drum_track.append(add_event)
                    total_ticks[9] = 0
                
                elif (evt.channel in melody_channels):   
                    if type(evt) == midi.events.NoteOnEvent:
                        add_event = midi.NoteOnEvent()
                    else:
                        add_event = midi.NoteOffEvent()

                    add_event.tick = total_ticks[evt.channel]
                    add_event.channel = evt.channel
                    add_event.data = evt.data

                    melody_track.append(add_event)
                    total_ticks[evt.channel] = 0
        
        eot = midi.EndOfTrackEvent(tick=1)
        drum_track.append(eot)
        melody_track.append(eot)

        if len(melody_track) > 10:
            melody_pattern.append(melody_track)
        if len(drum_track) > 10:
            drum_pattern.append(drum_track)

    return patternToStatematrix(melody_pattern), patternToStatematrix(drum_pattern)


def get_melody_and_bass(midifile):
    pattern = midi.read_midifile(midifile)
    
    melody_pattern = midi.Pattern()
    bass_pattern = midi.Pattern()

    melody_pattern.resolution = pattern.resolution
    bass_pattern.resolution = pattern.resolution
    
    for track in pattern:
        total_ticks = [0] * 17
    
        melody_track = midi.Track()
        bass_track = midi.Track()

        melody_channels = [0]
        bass_channels = [2] # For Nottingham database   
        for i in xrange(len(track)):
            evt = track[i]
            
            if isinstance(evt, midi.ProgramChangeEvent):
                if evt.data in [[0], [1], [2], [4], [5]]:
                    melody_channels.append(evt.channel)
                    melody_track.append(evt)
            
            if isinstance(evt, midi.ProgramChangeEvent):
                if evt.data in [[33], [34], [35], [36], [37], [38], [39], [40]]:
                    bass_channels.append(evt.channel)
                    bass_track.append(evt)

            elif isinstance(evt, midi.TimeSignatureEvent) or \
                    isinstance(evt, midi.SetTempoEvent) or \
                    isinstance(evt, midi.KeySignatureEvent):          
                melody_track.append(evt)
                bass_track.append(evt)
            
            elif isinstance(evt, midi.NoteEvent):
                total_ticks = [x + evt.tick for x in total_ticks]
            
                if (evt.channel in bass_channels):
                    if type(evt) == midi.events.NoteOnEvent:
                        add_event = midi.NoteOnEvent()
                    else:
                        add_event = midi.NoteOffEvent()

                    add_event.tick = total_ticks[evt.channel]
                    add_event.channel = evt.channel
                    add_event.data = evt.data

                    bass_track.append(add_event)
                    total_ticks[evt.channel] = 0
                
                elif (evt.channel in melody_channels):   
                    if type(evt) == midi.events.NoteOnEvent:
                        add_event = midi.NoteOnEvent()
                    else:
                        add_event = midi.NoteOffEvent()

                    add_event.tick = total_ticks[evt.channel]
                    add_event.channel = evt.channel
                    add_event.data = evt.data

                    melody_track.append(add_event)
                    total_ticks[evt.channel] = 0
        
        eot = midi.EndOfTrackEvent(tick=1)
        bass_track.append(eot)
        melody_track.append(eot)
        
        if len(melody_track) > 10:
            melody_pattern.append(melody_track)
        if len(bass_track) > 10:
            bass_pattern.append(bass_track)
            
    return patternToStatematrix(melody_pattern), patternToStatematrix(bass_pattern)
    

# Borrowed heavily from Daniel Johnson's midi manipulation code, with a few changes
# to shape of statematrix.
# Source: https://github.com/hexahedria/biaxial-rnn-music-composition/

def midiToStatematrix(midifile):
    pattern = midi.read_midifile(midifile)
    
    return patternToStatematrix(pattern)


def patternToStatematrix(pattern):
    timeleft = [0 for track in pattern]
    posns = [0 for track in pattern]

    statematrix = []
    time = 0

    state = [[0,0] for x in range(span)]
    statematrix.append(state)

    while True:
        if time % (pattern.resolution/4) == pattern.resolution/8:
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0],0] for x in range(span)]
            statematrix.append(state)
        for i in xrange(len(timeleft)): #For each track
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        # Ignore note outside range
                        pass
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-lowerBound] = [0, 0]
                        else:
                            state[evt.pitch-lowerBound] = [1, 1]
                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1
        
        if all(t is None for t in timeleft):
            break

        time += 1

    mat = np.array(statematrix)
    statematrix = np.hstack((mat[:, :, 0], mat[:, :, 1]))
    statematrix = np.asarray(statematrix).tolist()
    return statematrix


def statematrixToMidi(statematrix, name='test', bpm=120):
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    
    tempo = midi.SetTempoEvent(tick=0)
    tempo.set_bpm(bpm)
    track.append(tempo)
    
    tickscale = 55
    
    lastcmdtime = 0
    prevstate = [0 for i in range(span * 2)]
    for time, state in enumerate(statematrix):  
        offNotes = []
        onNotes = []
        for i in range(span):
            if prevstate[i] == 1:
                if state[i] == 0:
                    offNotes.append(i)
                elif state[i + span] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif state[i] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lowerBound))
            lastcmdtime = time
            
        prevstate = state
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)


if __name__ == '__main__':
    #preprocess('Classical_Music_Midi')
    
    for file in glob.glob('Classical_Music_Midi/*mid'):
        melody_mat, bass_mat = get_melody_and_bass(file) 
        
        if len(bass_mat) < 10:
            continue
        
        len_bass = len(bass_mat)
        len_melody = len(melody_mat)
        
        min_mat = melody_mat

        if(len_bass < len_melody):
            min_mat = bass_mat
            
        diff = abs(len_bass - len_melody)
        zeros = np.zeros((diff, 2*span))
        min_mat = np.vstack((min_mat, zeros))
            
        if(len_bass < len_melody):
            bass_mat = min_mat
        else:
            melody_mat = min_mat
        
        filename = file.split('/')[-1]
            
        write_filename = ''.join(filename.split('.')[:-1])
        print(write_filename)
        np.savetxt('Melody_Bass_Data/{}_melody.txt'.format(write_filename), melody_mat)
        np.savetxt('Melody_Bass_Data/{}_bass.txt'.format(write_filename), bass_mat)
        
