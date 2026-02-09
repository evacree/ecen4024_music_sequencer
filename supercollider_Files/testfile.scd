(
{ var ampOsc;
    ampOsc = SinOsc.kr(2, 1.5pi, 0.1, 0.5);
	SinOsc.ar([442, 440], 0, ampOsc);
}.play;
)

{ Pan2.ar(PinkNoise.ar(0.2), 0) }.play;

{SinOsc.ar(440, 0, 0.2) + Saw.ar(660, 0.2)}.play;

({
	var a, b, c, d;
	a = [SinOsc.ar(440, 0, 1), SinOsc.ar(442, 0, 1)];
	b = [SinOsc.ar(220, 0, 1), SinOsc.ar(221, 0, 1)];
	c = [SinOsc.ar(660, 0, 1), SinOsc.ar(663, 0, 1)];
	d = [SinOsc.ar(3520, 0, 0), SinOsc.ar(3520, 0, 0)];
	Mix([a, b, c, d]);
}.play;)

(
    var n = 20;
    {
        Mix.fill(n, { arg index;
            var freq;
            index.postln;
		freq = 440 + (index*440);
            freq.postln;
		SinOsc.ar(freq , 0, 1 / (n*n))
        })
    }.play;
)
x = { SinOsc.ar(660, 0, 0.2) }.play;
y = SynthDef.new("tutorial-SinOsc", { |out| Out.ar(out, SinOsc.ar(440, 0, 0.2)) }).play;
x.free;    // free just x
y.free;    // free just y

In.ar(0, 4); // this will return an Array of 4 OutputProxies

(
{
    var env = Env([0, 1, 0.5, 1, 0], [0.01, 0.5, 0.02, 0.5]);
    SinOsc.ar(470) * EnvGen.kr(env, doneAction: Done.freeSelf)
}.play
)


Env.perc(0.005, 0.5, 1, -4).plot;

{
	var a, b, c, d, env;
	env = Env.perc(0.008, 0.4).kr(doneAction: Done.freeSelf);
	a = [SinOsc.ar(440, 0, 1), SinOsc.ar(442, 0, 1)];
	b = [SinOsc.ar(220, 0, 1), SinOsc.ar(221, 0, 1)];
	c = [SinOsc.ar(660, 0, 1), SinOsc.ar(663, 0, 1)];
	d = [SinOsc.ar(3520, 0, 0.25), SinOsc.ar(3520, 0, 0.25)];
	Mix([a, b, c, d]) * env;
}.play;

SynthDef(\blip, { arg freq= 440, out= 0;
	var a, b, c, d, env, sig;
	env = Env.perc(0.008, 0.4).kr(doneAction: Done.freeSelf);
	a = [SinOsc.ar(freq, 0, 1), SinOsc.ar(freq+2, 0, 1)];
	b = [SinOsc.ar(freq/2, 0, 1), SinOsc.ar((freq+2)/2, 0, 1)];
	c = [SinOsc.ar((freq*3)/2, 0, 1), SinOsc.ar(((freq+2)*3)/2, 0, 1)];
	d = [SinOsc.ar(freq*8, 0, 0.25), SinOsc.ar((freq+2)*8, 0, 0.25)];
	sig = Mix([a, b, c, d]) * env;
	Out.ar(out, sig);
}).add;

Synth(\blip, ["freq", 660])

({
	var env, sig, trig;
	env = Env.perc(0.1, 0.2).kr(doneAction: Done.freeSelf);
	sig = [WhiteNoise.ar, WhiteNoise.ar]*env;
}.play;)

(
{
    var freq = EnvGen.kr(Env([1000, 50, 50], [0.01, 0.1], -4));
    var amp = EnvGen.kr(Env([0, 1, 0], [0.001, 0.3], -4), doneAction: 2);
    var sig = SinOsc.ar(freq) * amp;
    sig ! 2; // Stereo out
}.play;
)

SynthDef(\kik, { |basefreq = 20, ratio = 7, sweeptime = 0.05, preamp = 1, amp = 1,
        decay1 = 0.3, decay1L = 0.8, decay2 = 0.15, out = 0|
    var    fcurve = EnvGen.kr(Env([basefreq * ratio, basefreq], [sweeptime], \exp)),
        env = EnvGen.kr(Env([1, decay1L, 0], [decay1, decay2], -4), doneAction: Done.freeSelf),
        sig = SinOsc.ar(fcurve, 0.5pi, preamp).distort * env * amp;
    Out.ar(out, sig ! 2)
}).add;

Synth(\kik)

SynthDef(\kraftySnr, { |amp = 1, freq = 2000, rq = 3, decay = 0.3, pan, out|
    var    sig = PinkNoise.ar(amp),
        env = EnvGen.kr(Env.perc(0.01, decay), doneAction: Done.freeSelf);
    sig = BPF.ar(sig, freq, rq, env);
    Out.ar(out, Pan2.ar(sig, pan))
}).add;

Synth(\kraftySnr)

(
SynthDef(\snareClassic, {
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
)

Synth(\snareClassic);

(
SynthDef(\snareTight, {
    |out=0, amp=0.4, snap=8000|
    var env, noise, click, sig;

    env = EnvGen.kr(Env.perc(0.001, 0.15), doneAction: 2);

    noise = WhiteNoise.ar;
    noise = HPF.ar(noise, 3000);

    click = Decay2.ar(Impulse.ar(0), 0.001, 0.02) * SinOsc.ar(snap);

    sig = (noise + click) * env * amp;
    Out.ar(out, Pan2.ar(sig));
}).add;
)

Synth(\snareTight);

(
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
)

Synth(\snareDirty);

(
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
)

Synth(\kick808);


(
SynthDef(\kickHouse, {
    |out=0, amp=0.6, freq=60|
    var env, pitchEnv, sig;

    pitchEnv = EnvGen.kr(
        Env([freq * 2, freq], [0.01])
    );

    env = EnvGen.kr(
        Env.perc(0.001, 0.25),
        doneAction: 2
    );

    sig = SinOsc.ar(pitchEnv) * env;
    sig = sig + (sig * 0.3).tanh;

    Out.ar(out, Pan2.ar(sig * amp));
}).add;
)

Synth(\kickHouse);

(
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
)

Synth(\kickIndustrial);

MIDIClient.init;

MIDIIn.connectAll;

(
MIDIdef.noteOn(\noteOnTest, {
	arg v, note, chan, src;
	[v, note, chan, src].postln;
	~notes[note] = Synth.new(\tone, [
		\freq, note.midicps, \amp, v.linexp(1,127,0.01,0.3), \gate, 1

	]);
});
)

(
MIDIdef.noteOff(\noteOffTest, {
	arg v, note, chan, src;
	~notes[note].set(\gate, 0);
	~notes[note] = nil;
});
)

MIDIdef(\noteOnTest).free;
MIDIdef.freeAll;

(
SynthDef.new(\tone, {
	arg freq=440, amp=0.3, gate=0;
	var sig, env;
	sig = LFTri.ar(freq)!2;
	env = EnvGen.kr(Env.adsr, gate, doneAction:2);
	sig = sig*env*amp;
	Out.ar(0, sig);
}).add;
)

a = Synth.new(\tone, [\gate, 1]);
a.free;

~notes = Array.newClear(128);




(
SynthDef(\hammondKeyClick, { |out=0, amp=0.2|
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
)

Synth(\hammondKeyClick);

(
SynthDef(\hammondKeyClickVintage, { |out=0, amp=0.2|
    var env, noise, pop, click;

    env = EnvGen.kr(
        Env.perc(0.0003, 0.012, curve: -8),
        doneAction: 2
    );

    noise = WhiteNoise.ar;
    pop   = DC.ar(0.3); // contact transient

    click = (noise * 0.7 + pop) * env;

    click = HPF.ar(click, 1200);
    click = LPF.ar(click, 9000);

    Out.ar(out, (click * amp).dup);
}).add;
)

Synth(\hammondKeyClickVintage);

(
SynthDef(\acousticKick, {
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
)

Synth(\acousticKick);



























