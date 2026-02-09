(
//C2 == 36, C7 == 96
//C3 == 48, G5 == 79, C6 == 84, D#6 == 87, G6 == 91
s.boot;
MIDIClient.init;

MIDIIn.connectAll;

~bus = Bus.audio(s, 2);

~tables = Array.newClear(9);


~notes = Array.newClear(128);

~drawbars = Array.newClear(9);
~drawbars[0] = (8).linexp(0, 8, 0.00001, 0.3);
~drawbars[1] = (8).linexp(0, 8, 0.00001, 0.3);
~drawbars[2] = (8).linexp(0, 8, 0.00001, 0.3);
~drawbars[3] = (4).linexp(0, 8, 0.00001, 0.3);
~drawbars[4] = (0).linexp(0, 8, 0.00001, 0.3);
~drawbars[5] = (0).linexp(0, 8, 0.00001, 0.3);
~drawbars[6] = (0).linexp(0, 8, 0.00001, 0.3);
~drawbars[7] = (0).linexp(0, 8, 0.00001, 0.3);
~drawbars[8] = (5).linexp(0, 8, 0.00001, 0.3);

for(0, 8){
	arg i;
	~tables[i] = Buffer.alloc(s,2048);
};

//Lower Foldback
~tables[0].sine1([
	0, ~drawbars[2] + ~drawbars[0], ~drawbars[1],
	~drawbars[3], 0, ~drawbars[4], 0, ~drawbars[5],
	0, ~drawbars[6], 0, ~drawbars[7], 0, 0, 0, ~drawbars[8]
]);

//Main Register
~tables[1].sine1([
	~drawbars[0], ~drawbars[2], ~drawbars[1],
	~drawbars[3], 0, ~drawbars[4], 0, ~drawbars[5],
	0, ~drawbars[6], 0, ~drawbars[7], 0, 0, 0, ~drawbars[8]
]);

//3rd Octave Foldback
~tables[2].sine1([
	~drawbars[0], ~drawbars[2], ~drawbars[1],
	~drawbars[3], 0, ~drawbars[4], 0, ~drawbars[5] + ~drawbars[8],
	0, ~drawbars[6], 0, ~drawbars[7]
]);

//2nd Octave Fifth Foldback
~tables[3].sine1([
	~drawbars[0], ~drawbars[2], ~drawbars[1],
	~drawbars[3], 0, ~drawbars[4] + ~drawbars[7], 0,
	~drawbars[5] + ~drawbars[8], 0, ~drawbars[6]
]);

//2nd Octave Third Foldback
~tables[4].sine1([
	~drawbars[0], ~drawbars[2], ~drawbars[1],
	~drawbars[3], 0, ~drawbars[4] + ~drawbars[7], 0,
	~drawbars[5] + ~drawbars[8]
]);

//2nd Octave Foldback
~tables[5].sine1([
	~drawbars[0], ~drawbars[2], ~drawbars[1],
	~drawbars[3] + ~drawbars[5] + ~drawbars[8],
	0, ~drawbars[4] + ~drawbars[7]
]);

//1st Octave Fifth Foldback
~tables[6].sine1([
	~drawbars[0], ~drawbars[2],
	~drawbars[1] + ~drawbars[4] + ~drawbars[7],
	~drawbars[3] + ~drawbars[5] + ~drawbars[8]
]);


SynthDef(\leslieFX, { |inBus=0, out=0,
    slow=0.4,        // chorale speed (Hz)
    fast=6.0,        // tremolo speed (Hz)
    speed=0,         // 0 = slow, 1 = fast
    depth=0.003,     // pitch modulation depth (sec)
    tremDepth=0.6,   // amplitude modulation depth
    mix=0.6
|
    var input, rate, hornLFO, drumLFO;
    var hornDelay, trem, pan, wet;

    input = In.ar(inBus, 2);

    // smooth speed switching
    rate = Lag.kr(speed.linlin(0,1,slow,fast), 0.8);

    // horn (vibrato / pitch modulation)
    hornLFO = SinOsc.kr(rate, 0);
    hornDelay = DelayC.ar(
        input,
        0.05,
        0.02 + (hornLFO * depth)
    );

    // drum (amplitude modulation)
    drumLFO = SinOsc.kr(rate * 0.8, pi/2);
    trem = (1 - tremDepth) + (drumLFO * tremDepth);

    // stereo rotation
    pan = SinOsc.kr(rate, pi/2);
    wet = Balance2.ar(
        hornDelay[0],
        hornDelay[1],
        pan
    ) * trem;

    // dry / wet blend
    Out.ar(out, XFade2.ar(input, wet, mix.linlin(0,1,-1,1)));
}).add;


// Leslie
~leslie = Synth.after(1, \leslieFX, [
    \inBus, ~bus,
    \mix, 0.6
]);


MIDIdef.noteOn(\noteOnTest, {
	arg v, note, chan, src;
	[v, note, chan, src].postln;

	case(
		{(note >= 36) && (note < 48)},//Lower Foldback
		{~notes[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables[0],
			\gate, 1
		]);
		Synth.new(\keyClick, [\amp, 0.02]);
		},
		{(note >= 48) && (note <= 78)},//Main Register
		{~notes[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables[1],
			\gate, 1
		]);
		Synth.new(\keyClick, [\amp, 0.02]);
		},
		{(note > 78) && (note <= 83)},//3rd Octave Foldback
		{~notes[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables[2],
			\gate, 1
		]);
		Synth.new(\keyClick, [\amp, 0.02]);
		},
		{(note > 83) && (note <= 86)},//2nd Octave Fifth Foldback
		{~notes[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables[3],
			\gate, 1
		]);
		Synth.new(\keyClick, [\amp, 0.02]);
		},
		{(note > 86) && (note <= 90)},//2nd Octave Third Foldback
		{~notes[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables[4],
			\gate, 1
		]);
		Synth.new(\keyClick, [\amp, 0.02]);
		},
		{(note > 90) && (note <= 95)},//2nd Octave Foldback
		{~notes[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables[5],
			\gate, 1
		]);
		Synth.new(\keyClick, [\amp, 0.02]);
		},
		{note==96},//1st Octave Fifth Foldback
		{~notes[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables[6],
			\gate, 1
		]);
		Synth.new(\keyClick, [\amp, 0.02]);
		},
		{note==97},{Synth.new(\kickAcoustic);},
		{note==98},{Synth.new(\kick808);},
		{note==99},{Synth.new(\kickIndustrial);},
		{note==100},{Synth.new(\snare);},
		{note==101},{Synth.new(\snareClap);},
		{}
	);
});


MIDIdef.noteOff(\noteOffTest, {
	arg v, note, chan, src;
	~notes[note].set(\gate, -1);
	~notes[note] = nil;
});

SynthDef.new(\organ, {
	arg freq=440, table=~tables[1], gate=0;
	var sig, env;
	env = EnvGen.kr(Env.adsr, gate, doneAction:2);
	sig = Osc.ar(table, freq/2, mul:0.3!2);
	sig = sig*env;
	Out.ar(~bus, sig);
	//Out.ar(0, sig2);
}).add;

SynthDef(\keyClick, { |out=0, amp=0.2|
    var env, click;

    // very fast transient envelope
    env = EnvGen.kr(
        Env.perc(0.0005, 0.01, curve: -6),
        doneAction: 2
    );

    // broadband click source
    click = WhiteNoise.ar * env;

    // emphasize upper mids / highs (classic Hammond click range)
    click = BPF.ar(click, 3000, 0.7)
          + BPF.ar(click, 6000, 0.6);

    Out.ar(out, (click * amp).dup);
}).add;

SynthDef(\kickAcoustic, {
    |out=0, amp=0.8, freq=30, decay=0.35, clickAmp=0.3|

    var env, pitchEnv, body, click, kick;

    // Amplitude envelope
    env = EnvGen.kr(
        Env.perc(0.001, decay, curve: -4),
        doneAction: 2
    );

    // Pitch drop (essential for realism)
    pitchEnv = EnvGen.kr(
        Env([freq * 2.5, freq], [0.03], curve: -6)
    );

    // Body of the drum (air + shell)
    body = SinOsc.ar(pitchEnv) * env;

    // Beater click (very short noise burst)
    click = BPF.ar(
        WhiteNoise.ar(clickAmp),
        3000,
        0.7
    ) * EnvGen.kr(Env.perc(0.0005, 0.015));

    // Combine and gently saturate
    kick = (body + click) * amp;
    kick = tanh(kick * 2);

    Out.ar(out, kick ! 2);
}).add;

SynthDef(\kick808, {
    |out=0, amp=0.8, freq=30, decay=0.8|
    var env, pitchEnv, sig;

    pitchEnv = EnvGen.kr(
        Env([freq * 2, freq], [0.02]),
        doneAction: 0
    );

    env = EnvGen.kr(
        Env.perc(0.001, decay),
        doneAction: 2
    );

    sig = SinOsc.ar(pitchEnv) * env * amp;
    sig = LPF.ar(sig, 800);

    Out.ar(out, Pan2.ar(sig));
}).add;

SynthDef(\kickIndustrial, {
    |out=0, amp=0.7, freq=30, drive=6|
    var env, pitchEnv, sig;

    pitchEnv = EnvGen.kr(
        Env([freq * 4, freq], [0.02])
    );

    env = EnvGen.kr(
        Env.perc(0.001, 0.5),
        doneAction: 2
    );

    sig = SinOsc.ar(pitchEnv) * env;
    sig = tanh(sig * drive);

    sig = LPF.ar(sig, 1200);

    Out.ar(out, Pan2.ar(sig * amp));
}).add;

SynthDef(\snare, {
    |out=0, amp=0.5, toneFreq=180, noiseAmp=1, decay=0.25|
    var env, noise, tone, sig;

    env = EnvGen.kr(Env.perc(0.001, decay), doneAction: 2);

    noise = WhiteNoise.ar * noiseAmp;
    noise = BPF.ar(noise, 4000, 0.6);

    tone = SinOsc.ar(toneFreq, 0, 0.5);
    tone = tone * EnvGen.kr(Env.perc(0.001, decay * 0.7));

    sig = (noise + tone) * env * amp;
    sig = Pan2.ar(sig);

    Out.ar(out, sig);
}).add;

SynthDef(\snareClap, {
    |out=0, amp=0.5, drive=4|
    var env, noise, sig;

    env = EnvGen.kr(Env.perc(0.001, 0.3), doneAction: 2);

    noise = WhiteNoise.ar;
    noise = BPF.ar(noise, 3500, 0.8);

    sig = noise * env;
    sig = tanh(sig * drive) * amp;

    Out.ar(out, Pan2.ar(sig));
}).add;


~leslie.set(\speed, 0);

)