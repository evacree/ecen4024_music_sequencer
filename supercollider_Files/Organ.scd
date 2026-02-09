(
s.boot;
MIDIClient.init;

MIDIIn.connectAll;

~bus = Bus.audio(s, 2);

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


MIDIdef.noteOn(\noteOnTest, {
	arg v, note, chan, src;
	[v, note, chan, src].postln;
	~notes[note] = Synth.new(\organ, [
		\freq, note.midicps,
		//\amp, v.linexp(1,127,0.01,0.3),
		\gate, 1
	]);
	Synth.new(\keyClick, [\amp, 0.02]);
});


MIDIdef.noteOff(\noteOffTest, {
	arg v, note, chan, src;
	~notes[note].set(\gate, -1);
	~notes[note] = nil;
});


SynthDef.new(\tone, {
	arg freq=440, amp=0.3, gate=0;
	var sig, env;
	sig = LFTri.ar(freq)!2;
	env = EnvGen.kr(Env.adsr, gate, doneAction:2);
	sig = sig*env*amp;
	Out.ar(0, sig);
}).add;

SynthDef.new(\organ, {
	arg freq=440, gate=0;
	var sig0, sig1, sig2, sig3, sig4, sig5,
	sig6, sig7, sig8, env, main;
	env = EnvGen.kr(Env.adsr, gate, doneAction:2);
	sig0 = LPF.ar(SinOsc.ar(freq*0.5), 8000)!2;
	sig0 = sig0*env*~drawbars[0];
	sig1 = LPF.ar(SinOsc.ar(freq*1.5), 8000)!2;
	sig1 = sig1*env*~drawbars[1];
	sig2 = LPF.ar(SinOsc.ar(freq), 8000)!2;
	sig2 = sig2*env*~drawbars[2];
	sig3 = LPF.ar(SinOsc.ar(freq*2), 8000)!2;
	sig3 = sig3*env*~drawbars[3];
	sig4 = LPF.ar(SinOsc.ar(freq*3), 8000)!2;
	sig4 = sig4*env*~drawbars[4];
	sig5 = LPF.ar(SinOsc.ar(freq*4), 8000)!2;
	sig5 = sig5*env*~drawbars[5];
	sig6 = LPF.ar(SinOsc.ar(freq*5), 8000)!2;
	sig6 = sig6*env*~drawbars[6];
	sig7 = LPF.ar(SinOsc.ar(freq*6), 8000)!2;
	sig7 = sig7*env*~drawbars[7];
	sig8 = LPF.ar(SinOsc.ar(freq*8), 8000)!2;
	sig8 = sig8*env*~drawbars[8];
	Out.ar(~bus, Mix.new([sig0, sig1 ,sig2, sig3,
		sig4, sig5, sig6, sig7, sig8
	]));
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


SynthDef(\chorusFX, { |inBus=0, out=0,
    mix=0.4,        // dry/wet
    depth=0.004,    // modulation depth (seconds)
    rate=0.3,       // modulation rate (Hz)
    delay=0.02      // base delay (seconds)
|
    var input, chorus, lfos, delays;

    input = In.ar(inBus, 2);

    // three independent LFOs for richness
    lfos = [
        SinOsc.kr(rate, 0),
        SinOsc.kr(rate * 1.13, 2),
        SinOsc.kr(rate * 0.87, 4)
    ];

    delays = lfos.collect { |lfo|
        DelayC.ar(
            input,
            0.05,
            delay + (lfo * depth)
        )
    };

    chorus = Mix(delays) * 0.6;

    // dry/wet blend
    Out.ar(out, XFade2.ar(input, chorus, mix.linlin(0,1,-1,1)));
}).add;


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

~leslie.set(\speed, 0);

)