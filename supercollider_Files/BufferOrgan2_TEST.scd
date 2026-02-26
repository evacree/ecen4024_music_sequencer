(
// ============================
// BufferOrgan2 (PD -> SC MIDI)
// with MIDI input selector window
// ============================
//
// Notes reference:
// C2 = 36, C3 = 48, C4 = 60, C7 = 96

s.quit;  // safe if already running

s.options.outDevice = "Speakers (Realtek(R) Audio)";
s.options.numOutputBusChannels = 2;

s.boot;

s.waitForBoot {

    // Declare locals at top of function (required in SC)
    var w, label;

    // ---- MIDI setup ----
    MIDIClient.init;

    "=== MIDI SOURCES ===".postln;
    MIDIClient.sources.do { |src, i|
        [i, src.device, src.name].postln;
    };

    // Utility: connect ONLY to one MIDI source index
    ~connectMidiSource = { |idx|
        MIDIIn.disconnectAll;
        MIDIIn.connect(idx);
        ("Connected MIDI input ONLY to source " ++ idx ++ ": " ++ MIDIClient.sources[idx].name).postln;
    };

    // Utility: find index by exact name
    ~midiIndexByName = { |name|
        MIDIClient.sources.detectIndex { |src| src.name.asString == name.asString };
    };

    ~mpName = "MediaPipe_to_PureData";
    ~pdName = "PureData_SuperCollider";

    ~mpIndex = ~midiIndexByName.(~mpName);
    ~pdIndex = ~midiIndexByName.(~pdName);

    if(~mpIndex.isNil or: ~pdIndex.isNil) {
        "ERROR: Could not find one or both MIDI input ports by name.".postln;
        ("Need: " ++ ~mpName ++ " and " ++ ~pdName).postln;
        "Available sources:".postln;
        MIDIClient.sources.do { |src, i| [i, src.device, src.name].postln; };
    };

    // ---- Create MIDI Source Select window ----
    w = Window("MIDI Input Select (SuperCollider)", Rect(200, 200, 380, 120)).front.alwaysOnTop_(true);

    label = StaticText(w, Rect(10, 10, 360, 20))
        .string_("Default: PureData_SuperCollider (Sequencer)");

    // Button: listen to sequencer (PD -> SC)
    Button(w, Rect(10, 40, 360, 28))
        .states_([["Listen: PureData_SuperCollider (Sequencer)", Color.black, Color.green]])
        .action_({
            if(~pdIndex.notNil) {
                ~connectMidiSource.(~pdIndex);
                label.string = "Listening: PureData_SuperCollider (Sequencer)";
            }{
                label.string = "ERROR: PureData_SuperCollider not found.";
            };
        });

    // Button: listen to gesture stream (MediaPipe -> PD port)
    Button(w, Rect(10, 75, 360, 28))
        .states_([["Listen: MediaPipe_to_PureData (Gestures)", Color.black, Color.yellow]])
        .action_({
            if(~mpIndex.notNil) {
                ~connectMidiSource.(~mpIndex);
                label.string = "Listening: MediaPipe_to_PureData (Gestures)";
            }{
                label.string = "ERROR: MediaPipe_to_PureData not found.";
            };
        });

    // Default: listen to sequencer
    if(~pdIndex.notNil) {
        ~connectMidiSource.(~pdIndex);
    };

    // ---- Audio routing ----
    ~bus = Bus.audio(s, 2);

    // ---- Wavetables ----
    ~tables = Array.newClear(9);
    for(0, 8) { |i|
        ~tables[i] = Buffer.alloc(s, 2048);
    };

    // ---- Per-note synth storage ----
    ~notes = Array.newClear(128);

    // ---- Drawbar settings ----
    ~settings = Array.newClear(11);
    ~drawbars = Array.newClear(9);
    ~highSetting = 0;

    ~updateDrawbars = {
        if(~settings[0] == 0){~drawbars[0] = 0}{ ~drawbars[0] = (~settings[0]).linexp(0, 8, 0.00001, 0.3); };
        if(~settings[1] == 0){~drawbars[1] = 0}{ ~drawbars[1] = (~settings[1]).linexp(0, 8, 0.00001, 0.3); };
        if(~settings[2] == 0){~drawbars[2] = 0}{ ~drawbars[2] = (~settings[2]).linexp(0, 8, 0.00001, 0.3); };
        if(~settings[3] == 0){~drawbars[3] = 0}{ ~drawbars[3] = (~settings[3]).linexp(0, 8, 0.00001, 0.3); };
        if(~settings[4] == 0){~drawbars[4] = 0}{ ~drawbars[4] = (~settings[4]).linexp(0, 8, 0.00001, 0.3); };
        if(~settings[5] == 0){~drawbars[5] = 0}{ ~drawbars[5] = (~settings[5]).linexp(0, 8, 0.00001, 0.3); };
        if(~settings[6] == 0){~drawbars[6] = 0}{ ~drawbars[6] = (~settings[6]).linexp(0, 8, 0.00001, 0.3); };
        if(~settings[7] == 0){~drawbars[7] = 0}{ ~drawbars[7] = (~settings[7]).linexp(0, 8, 0.00001, 0.3); };
        if(~settings[8] == 0){~drawbars[8] = 0}{ ~drawbars[8] = (~settings[8]).linexp(0, 8, 0.00001, 0.3); };
    };

    ~updateTables = {
        ~tables[0].sine1([ 0, ~drawbars[2] + ~drawbars[0], ~drawbars[1], ~drawbars[3], 0, ~drawbars[4], 0, ~drawbars[5], 0, ~drawbars[6], 0, ~drawbars[7], 0, 0, 0, ~drawbars[8] ]);
        ~tables[1].sine1([ ~drawbars[0], ~drawbars[2], ~drawbars[1], ~drawbars[3], 0, ~drawbars[4], 0, ~drawbars[5], 0, ~drawbars[6], 0, ~drawbars[7], 0, 0, 0, ~drawbars[8] ]);
        ~tables[2].sine1([ ~drawbars[0], ~drawbars[2], ~drawbars[1], ~drawbars[3], 0, ~drawbars[4], 0, ~drawbars[5] + ~drawbars[8], 0, ~drawbars[6], 0, ~drawbars[7] ]);
        ~tables[3].sine1([ ~drawbars[0], ~drawbars[2], ~drawbars[1], ~drawbars[3], 0, ~drawbars[4] + ~drawbars[7], 0, ~drawbars[5] + ~drawbars[8], 0, ~drawbars[6] ]);
        ~tables[4].sine1([ ~drawbars[0], ~drawbars[2], ~drawbars[1], ~drawbars[3], 0, ~drawbars[4] + ~drawbars[7], 0, ~drawbars[5] + ~drawbars[8] ]);
        ~tables[5].sine1([ ~drawbars[0], ~drawbars[2], ~drawbars[1], ~drawbars[3] + ~drawbars[5] + ~drawbars[8], 0, ~drawbars[4] + ~drawbars[7] ]);
        ~tables[6].sine1([ ~drawbars[0], ~drawbars[2], ~drawbars[1] + ~drawbars[4] + ~drawbars[7], ~drawbars[3] + ~drawbars[5] + ~drawbars[8] ]);
    };

    ~changeSettings = { |setting=0, increment=1|
        if(increment == 0) { increment = -1 };
        if((setting >= 0) && (setting < 9)) {
            if((increment == 1) && (~settings[setting] < 8)) { ~settings[setting] = ~settings[setting] + 1; };
            if((increment == -1) && (~settings[setting] > 0)) { ~settings[setting] = ~settings[setting] - 1; };
        };
        ~settings.postln;
        ~updateDrawbars.();
        ~updateTables.();
    };

    // ---- Initialize drawbar defaults ----
    ~settings[0] = 8; ~settings[1] = 8; ~settings[2] = 8; ~settings[3] = 4;
    ~settings[4] = 0; ~settings[5] = 0; ~settings[6] = 0; ~settings[7] = 0; ~settings[8] = 4;
    ~updateDrawbars.();
    ~updateTables.();

    // ---- SynthDefs ----
    SynthDef(\leslieFX, { |inBus=0, out=0, slow=0.4, fast=6.0, speed=0, depth=0.003, tremDepth=0.6, mix=0.6|
        var input, rate, hornLFO, drumLFO, hornDelay, trem, pan, wet;
        input = In.ar(inBus, 2);
        rate = Lag.kr(speed.linlin(0, 1, slow, fast), 0.8);
        hornLFO = SinOsc.kr(rate, 0);
        hornDelay = DelayC.ar(input, 0.05, 0.02 + (hornLFO * depth));
        drumLFO = SinOsc.kr(rate * 0.8, pi/2);
        trem = (1 - tremDepth) + (drumLFO * tremDepth);
        pan = SinOsc.kr(rate, pi/2);
        wet = Balance2.ar(hornDelay[0], hornDelay[1], pan) * trem;
        Out.ar(out, XFade2.ar(input, wet, mix.linlin(0, 1, -1, 1)));
    }).add;

    SynthDef(\organ, { |freq=440, table=0, gate=0|
        var sig, env;
        env = EnvGen.kr(Env.adsr, gate, doneAction: 2);
        sig = Osc.ar(table, freq/2, mul: 0.3 ! 2);
        sig = sig * env;
        Out.ar(~bus, sig);
    }).add;

    SynthDef(\keyClick, { |out=0, amp=0.2|
        var env, click;
        env = EnvGen.kr(Env.perc(0.0005, 0.01, curve: -6), doneAction: 2);
        click = WhiteNoise.ar * env;
        click = BPF.ar(click, 3000, 0.7) + BPF.ar(click, 6000, 0.6);
        Out.ar(out, (click * amp).dup);
    }).add;

    // ---- Start Leslie FX ----
    ~leslie = Synth.tail(s, \leslieFX, [\inBus, ~bus, \mix, 0.6, \out, 0]);
    ~leslie.set(\speed, 0);

    // ==========================================
    // MIDI handlers (nil-safe)
    // ==========================================
    MIDIdef.noteOn(\noteOnFromSelectedInput, { |vel, note, chan, src|
        [\noteOn, vel, note, chan, src].postln;

        if(~notes[note].notNil) { ~notes[note].set(\gate, -1); };

        // PD channel 1 -> SC chan 0
        if(chan == 0) {
            case(
                {(note >= 36) && (note < 48)}, { ~notes[note] = Synth(\organ, [\freq, note.midicps, \table, ~tables[0], \gate, 1]); Synth(\keyClick, [\amp, 0.02]); },
                {(note >= 48) && (note <= 78)}, { ~notes[note] = Synth(\organ, [\freq, note.midicps, \table, ~tables[1], \gate, 1]); Synth(\keyClick, [\amp, 0.02]); },
                {(note > 78) && (note <= 83)}, { ~notes[note] = Synth(\organ, [\freq, note.midicps, \table, ~tables[2], \gate, 1]); Synth(\keyClick, [\amp, 0.02]); },
                {(note > 83) && (note <= 86)}, { ~notes[note] = Synth(\organ, [\freq, note.midicps, \table, ~tables[3], \gate, 1]); Synth(\keyClick, [\amp, 0.02]); },
                {(note > 86) && (note <= 90)}, { ~notes[note] = Synth(\organ, [\freq, note.midicps, \table, ~tables[4], \gate, 1]); Synth(\keyClick, [\amp, 0.02]); },
                {(note > 90) && (note <= 95)}, { ~notes[note] = Synth(\organ, [\freq, note.midicps, \table, ~tables[5], \gate, 1]); Synth(\keyClick, [\amp, 0.02]); },
                {note == 96}, { ~notes[note] = Synth(\organ, [\freq, note.midicps, \table, ~tables[6], \gate, 1]); Synth(\keyClick, [\amp, 0.02]); },
                {note == 97}, { ~changeSettings.(0, ~highSetting); },
                {note == 98}, { ~changeSettings.(1, ~highSetting); },
                {note == 99}, { ~changeSettings.(2, ~highSetting); },
                {note == 100}, { ~changeSettings.(3, ~highSetting); },
                {note == 101}, { ~changeSettings.(4, ~highSetting); },
                {note == 102}, { ~changeSettings.(5, ~highSetting); },
                {note == 103}, { ~changeSettings.(6, ~highSetting); },
                {note == 104}, { ~changeSettings.(7, ~highSetting); },
                {note == 105}, { ~changeSettings.(8, ~highSetting); },
                {note == 106}, { ~changeSettings.(9, ~highSetting); },
                {note == 107}, { ~changeSettings.(10, ~highSetting); },
                {note == 108}, { ~highSetting = 1; },
                {true}, { }
            );
        };
    });

    MIDIdef.noteOff(\noteOffFromSelectedInput, { |vel, note, chan, src|
        [\noteOff, vel, note, chan, src].postln;

        if(~notes[note].notNil) {
            ~notes[note].set(\gate, -1);
            ~notes[note] = nil;
        };

        if(note == 108) { ~highSetting = 0; };
    });

    "SuperCollider organ ready. Use the window to select MIDI input (default = PureData_SuperCollider).".postln;
};
)