(
//C2 == 36, C7 == 96
//C3 == 48, G5 == 79, C6 == 84, D#6 == 87, G6 == 91
s.boot;
MIDIClient.init;

//~port = MIDIClient.sources.detect { |src|
//    src.device.contains("SynthPort")
//};
//MIDIIn.connect(0, ~port);

MIDIIn.connectAll;

~amp = 1;

~bus0a = Bus.audio(s, 2);
~bus1a = Bus.audio(s, 2);
~bus2a = Bus.audio(s, 2);
~bus3a = Bus.audio(s, 2);

~bus0b = Bus.audio(s, 2);
~bus1b = Bus.audio(s, 2);


~tables = Array.newClear(9);
~tables2 = Array.newClear(9);

for(0, 8){
	arg i;
	~tables[i] = Buffer.alloc(s,2048);
	~tables2[i] = Buffer.alloc(s,2048);
};

~notes = Array.newClear(128);
~notes2 = Array.newClear(128);
~notes3 = Array.newClear(128);
~notes4 = Array.newClear(128);

~settings = Array.newClear(11);
~drawbars = Array.newClear(9);
~highSetting = 0;

~settings2 = Array.newClear(11);
~drawbars2 = Array.newClear(9);
~highSetting2 = 0;

~bassType = 0;

~updateDrawbars = {
	if(~settings[0] == 0){~drawbars[0] = 0}{
		~drawbars[0] = (~settings[0]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings[1] == 0){~drawbars[1] = 0}{
		~drawbars[1] = (~settings[1]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings[2] == 0){~drawbars[2] = 0}{
		~drawbars[2] = (~settings[2]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings[3] == 0){~drawbars[3] = 0}{
		~drawbars[3] = (~settings[3]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings[4] == 0){~drawbars[4] = 0}{
		~drawbars[4] = (~settings[4]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings[5] == 0){~drawbars[5] = 0}{
		~drawbars[5] = (~settings[5]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings[6] == 0){~drawbars[6] = 0}{
		~drawbars[6] = (~settings[6]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings[7] == 0){~drawbars[7] = 0}{
		~drawbars[7] = (~settings[7]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings[8] == 0){~drawbars[8] = 0}{
		~drawbars[8] = (~settings[8]).linexp(0, 8, 0.00001, 0.3);
	};
};

~updateDrawbars2 = {
	if(~settings2[0] == 0){~drawbars2[0] = 0}{
		~drawbars2[0] = (~settings2[0]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings2[1] == 0){~drawbars2[1] = 0}{
		~drawbars2[1] = (~settings2[1]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings2[2] == 0){~drawbars2[2] = 0}{
		~drawbars2[2] = (~settings2[2]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings2[3] == 0){~drawbars2[3] = 0}{
		~drawbars2[3] = (~settings2[3]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings2[4] == 0){~drawbars2[4] = 0}{
		~drawbars2[4] = (~settings2[4]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings2[5] == 0){~drawbars2[5] = 0}{
		~drawbars2[5] = (~settings2[5]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings2[6] == 0){~drawbars2[6] = 0}{
		~drawbars2[6] = (~settings2[6]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings2[7] == 0){~drawbars2[7] = 0}{
		~drawbars2[7] = (~settings2[7]).linexp(0, 8, 0.00001, 0.3);
	};
	if(~settings2[8] == 0){~drawbars2[8] = 0}{
		~drawbars2[8] = (~settings2[8]).linexp(0, 8, 0.00001, 0.3);
	};
};

~updateTables = {
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
};

~updateTables2 = {
	//Lower Foldback
	~tables2[0].sine1([
		0, ~drawbars2[2] + ~drawbars2[0], ~drawbars2[1],
		~drawbars2[3], 0, ~drawbars2[4], 0, ~drawbars2[5],
		0, ~drawbars2[6], 0, ~drawbars2[7], 0, 0, 0, ~drawbars2[8]
	]);

	//Main Register
	~tables2[1].sine1([
		~drawbars2[0], ~drawbars2[2], ~drawbars2[1],
		~drawbars2[3], 0, ~drawbars2[4], 0, ~drawbars2[5],
		0, ~drawbars2[6], 0, ~drawbars2[7], 0, 0, 0, ~drawbars2[8]
	]);

	//3rd Octave Foldback
	~tables2[2].sine1([
		~drawbars2[0], ~drawbars2[2], ~drawbars2[1],
		~drawbars2[3], 0, ~drawbars2[4], 0, ~drawbars2[5] + ~drawbars2[8],
		0, ~drawbars2[6], 0, ~drawbars2[7]
	]);

	//2nd Octave Fifth Foldback
	~tables2[3].sine1([
		~drawbars2[0], ~drawbars2[2], ~drawbars2[1],
		~drawbars2[3], 0, ~drawbars2[4] + ~drawbars2[7], 0,
		~drawbars2[5] + ~drawbars2[8], 0, ~drawbars2[6]
	]);

	//2nd Octave Third Foldback
	~tables2[4].sine1([
		~drawbars2[0], ~drawbars2[2], ~drawbars2[1],
		~drawbars2[3], 0, ~drawbars2[4] + ~drawbars2[7], 0,
		~drawbars2[5] + ~drawbars2[8]
	]);

	//2nd Octave Foldback
	~tables2[5].sine1([
		~drawbars2[0], ~drawbars2[2], ~drawbars2[1],
		~drawbars2[3] + ~drawbars2[5] + ~drawbars2[8],
		0, ~drawbars2[4] + ~drawbars2[7]
	]);

	//1st Octave Fifth Foldback
	~tables2[6].sine1([
		~drawbars2[0], ~drawbars2[2],
		~drawbars2[1] + ~drawbars2[4] + ~drawbars2[7],
		~drawbars2[3] + ~drawbars2[5] + ~drawbars2[8]
	]);
};

~changeSettings = {
	arg setting = 0, increment = 1, chan = 0;
	if(increment == 0){increment = -1}{};
	if(chan == 0){
	case(
			{(setting >= 0) && (setting <9)},
			{if((increment == 1) && (~settings[setting] < 8))
				{~settings[setting] = ~settings[setting]+1;}
				{if((increment == -1) && (~settings[setting] > 0))
					{~settings[setting] = ~settings[setting]-1;}{}
				}
			},
			{}
		);
		~settings.postln;
		~updateDrawbars.();
		~updateTables.();
	}{if(chan == 1){
	case(
			{(setting >= 0) && (setting <9)},
			{if((increment == 1) && (~settings2[setting] < 8))
				{~settings2[setting] = ~settings2[setting]+1;}
				{if((increment == -1) && (~settings2[setting] > 0))
					{~settings2[setting] = ~settings2[setting]-1;}{}
				}
			},
			{}
		);
		~settings2.postln;
		~updateDrawbars2.();
		~updateTables2.();
	}{}
	};
};

~settings[0] = 8;
~settings[1] = 8;
~settings[2] = 8;
~settings[3] = 4;
~settings[4] = 0;
~settings[5] = 0;
~settings[6] = 0;
~settings[7] = 0;
~settings[8] = 4;
~updateDrawbars.();
~updateTables.();

~settings2[0] = 8;
~settings2[1] = 8;
~settings2[2] = 8;
~settings2[3] = 8;
~settings2[4] = 8;
~settings2[5] = 8;
~settings2[6] = 8;
~settings2[7] = 8;
~settings2[8] = 8;
~updateDrawbars2.();
~updateTables2.();


~ch0Function = {
	arg v = 100, note = 60, chan = 0, src = 1;
	~notes[note].set(\gate, -1);
	case(
		{(note >= 36) && (note < 48)},//Lower Foldback
		{~notes[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables[0],
			\gate, 1,
			\out, ~bus0a
		]);
		Synth.new(\keyClick, [\out, ~bus0a, \amp, 0.02]);
		},
		{(note >= 48) && (note <= 78)},//Main Register
		{~notes[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables[1],
			\gate, 1,
			\out, ~bus0a
		]);
		Synth.new(\keyClick, [\out, ~bus0a, \amp, 0.02]);
		},
		{(note > 78) && (note <= 83)},//3rd Octave Foldback
		{~notes[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables[2],
			\gate, 1,
			\out, ~bus0a
		]);
		Synth.new(\keyClick, [\out, ~bus0a, \amp, 0.02]);
		},
		{(note > 83) && (note <= 86)},//2nd Octave Fifth Foldback
		{~notes[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables[3],
			\gate, 1,
			\out, ~bus0a
		]);
		Synth.new(\keyClick, [\out, ~bus0a, \amp, 0.02]);
		},
		{(note > 86) && (note <= 90)},//2nd Octave Third Foldback
		{~notes[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables[4],
			\gate, 1,
			\out, ~bus0a
		]);
		Synth.new(\keyClick, [\out, ~bus0a, \amp, 0.02]);
		},
		{(note > 90) && (note <= 95)},//2nd Octave Foldback
		{~notes[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables[5],
			\gate, 1,
			\out, ~bus0a
		]);
		Synth.new(\keyClick, [\out, ~bus0a, \amp, 0.02]);
		},
		{note==96},//1st Octave Fifth Foldback
		{~notes[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables[6],
			\gate, 1,
			\out, ~bus0a
		]);
		Synth.new(\keyClick, [\out, ~bus0a, \amp, 0.02]);
		},
		{note==97},{~changeSettings.(0, ~highSetting);},
		{note==98},{~changeSettings.(1, ~highSetting);},
		{note==99},{~changeSettings.(2, ~highSetting);},
		{note==100},{~changeSettings.(3, ~highSetting);},
		{note==101},{~changeSettings.(4, ~highSetting);},
		{note==102},{~changeSettings.(5, ~highSetting);},
		{note==103},{~changeSettings.(6, ~highSetting);},
		{note==104},{~changeSettings.(7, ~highSetting);},
		{note==105},{~changeSettings.(8, ~highSetting);},
		{note==106},{~changeSettings.(9, ~highSetting);},
		{note==107},{~changeSettings.(10, ~highSetting);},
		{note==108},{~highSetting = 1;},
		{}
	);
};

~ch1Function = {
	arg v = 100, note = 60, chan = 1, src = 1;
	~notes2[note].set(\gate, -1);
	case(
		{(note >= 36) && (note < 48)},//Lower Foldback
		{~notes2[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables2[0],
			\gate, 1,
			\out, ~bus1a
		]);
		Synth.new(\keyClick, [\out, ~bus1a, \amp, 0.02]);
		},
		{(note >= 48) && (note <= 78)},//Main Register
		{~notes2[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables2[1],
			\gate, 1,
			\out, ~bus1a
		]);
		Synth.new(\keyClick, \out, ~bus1a, [\amp, 0.02]);
		},
		{(note > 78) && (note <= 83)},//3rd Octave Foldback
		{~notes2[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables2[2],
			\gate, 1,
			\out, ~bus1a
		]);
		Synth.new(\keyClick, [\out, ~bus1a, \amp, 0.02]);
		},
		{(note > 83) && (note <= 86)},//2nd Octave Fifth Foldback
		{~notes2[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables2[3],
			\gate, 1,
			\out, ~bus1a
		]);
		Synth.new(\keyClick, [\out, ~bus1a, \amp, 0.02]);
		},
		{(note > 86) && (note <= 90)},//2nd Octave Third Foldback
		{~notes2[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables2[4],
			\gate, 1,
			\out, ~bus1a
		]);
		Synth.new(\keyClick, [\out, ~bus1a, \amp, 0.02]);
		},
		{(note > 90) && (note <= 95)},//2nd Octave Foldback
		{~notes2[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables2[5],
			\gate, 1,
			\out, ~bus1a
		]);
		Synth.new(\keyClick, [\out, ~bus1a, \amp, 0.02]);
		},
		{note==96},//1st Octave Fifth Foldback
		{~notes2[note] = Synth.new(\organ, [
			\freq, note.midicps,
			\table, ~tables2[6],
			\gate, 1,
			\out, ~bus1a
		]);
		Synth.new(\keyClick, [\out, ~bus1a, \amp, 0.02]);
		},
		{note==97},{~changeSettings.(0, ~highSetting2, 0);},
		{note==98},{~changeSettings.(1, ~highSetting2, 0);},
		{note==99},{~changeSettings.(2, ~highSetting2, 0);},
		{note==100},{~changeSettings.(3, ~highSetting2, 0);},
		{note==101},{~changeSettings.(4, ~highSetting2, 0);},
		{note==102},{~changeSettings.(5, ~highSetting2, 0);},
		{note==103},{~changeSettings.(6, ~highSetting2, 0);},
		{note==104},{~changeSettings.(7, ~highSetting2, 0);},
		{note==105},{~changeSettings.(8, ~highSetting2, 0);},
		{note==106},{~changeSettings.(9, ~highSetting2, 0);},
		{note==107},{~changeSettings.(10, ~highSetting2, 0);},
		{note==108},{~highSetting2 = 1;},
		{}
	);
};

~ch2Function = {
	arg v = 100, note = 60, chan = 2, src = 1;
	~notes3[note].set(\gate, -1);
	case(
		{(note >= 36) && (note <= 96) && (~bassType == 0)},//Synth Bass
		{~notes3[note] = Synth.new(\synthBass, [
			\out, ~bus2a,
			\freq, (note.midicps)/2,
			\gate, 1
		]);
		},
		{(note >= 36) && (note <= 96) && (~bassType == 1)},//String Bass
		{~notes3[note] = Synth.new(\doubleBass, [
			\out, ~bus2a,
			\freq, (note.midicps)/2,
			\gate, 1
		]);
		},
		{note==97},{~bassType = 0;},
		{note==98},{~bassType = 1;},
		{note==99},{},
		{note==100},{},
		{note==101},{},
		{note==102},{},
		{note==103},{},
		{note==104},{},
		{note==105},{},
		{note==106},{},
		{note==107},{},
		{note==108},{},
		{}
	);

};

~ch3Function = {
	arg v = 100, note = 60, chan = 2, src = 1;
	~notes4[note].set(\gate, -1);
	case(
		{note==60},{~notes4[note] = Synth.new(\kickAcoustic, [\out, ~bus3a]);},
		{note==61},{~notes4[note] = Synth.new(\kick808, [\out, ~bus3a]);},
		{note==62},{~notes4[note] = Synth.new(\kickIndustrial, [\out, ~bus3a]);},
		{note==63},{~notes4[note] = Synth.new(\snare, [\out, ~bus3a]);},
		{note==64},{~notes4[note] = Synth.new(\snareClap, [\out, ~bus3a]);},
		{note==65},{},
		{note==66},{},
		{note==67},{},
		{note==68},{},
		{note==69},{},
		{note==70},{},
		{note==71},{},
		{note==72},{},
		{}
	);

};


MIDIdef.noteOn(\noteOnTest, {
	arg v, note, chan, src;
	[v, note, chan, src, "IN"].postln;
	if(chan == 0){
		~ch0Function.(v, note, chan, src);
	}{if(chan == 1){
		~ch1Function.(v, note, chan, src);
	}{}};
	if(chan == 2){
		~ch2Function.(v, note, chan, src);
	}{if(chan == 3){
		~ch3Function.(v, note, chan, src);
	}{}};
});


//{note==97},{Synth.new(\kickAcoustic);},
//{note==98},{Synth.new(\kick808);},
//{note==99},{Synth.new(\kickIndustrial);},
//{note==100},{Synth.new(\snare);},
//{note==101},{Synth.new(\snareClap);},


MIDIdef.noteOff(\noteOffTest, {
	arg v, note, chan, src;
	["000", note, chan, src, "OFF"].postln;
	if(chan == 0){
		~notes[note].set(\gate, -1);
		~notes[note] = nil;
		if(note==108){~highSetting = 0;}{};
	}{if(chan == 1){
		~notes2[note].set(\gate, -1);
		~notes2[note] = nil;
		if(note==108){~highSetting2 = 0;}{};
	}{}};
	if(chan == 2){
		~notes3[note].set(\gate, 0);
		~notes3[note] = nil;
	}{if(chan == 3){
		~notes4[note] = nil;
	}{}};
});

MIDIdef.bend(\volControl, { |val, chan, src|
	var pos = val.linlin(0, 16383, 0, 1);
	~mixer.set(\amp, pos);
	[pos].postln;
});

// Leslie
~leslie = Synth.after(1, \leslieFX, [
    \inBus, ~bus0a,
	\out, ~bus0b,
    \mix, 0.6
]);

~leslie2 = Synth.after(1, \leslieFX, [
    \inBus, ~bus1a,
	\out, ~bus1b,
    \mix, 0.6
]);

~mixer = Synth.after(~leslie, \mixer, [
	\bus0, ~bus0b,
	\bus1, ~bus1b,
	\bus2, ~bus2a,
	\bus3, ~bus3a
]);

~leslie.set(\mix, 0.6);


SynthDef.new(\organ, {
	arg freq=440, table=~tables[1], gate=0, out=0;
	var sig, env;
	env = EnvGen.kr(Env.adsr, gate, doneAction:2);
	sig = Osc.ar(table, freq/2, mul:0.3!2);
	sig = sig*env;
	Out.ar(out, sig);
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

SynthDef(\leslieFX, { |inBus=0, out=0,
    slow=0.4,        // chorale speed (Hz)
    fast=6.0,        // tremolo speed (Hz)
    speed=0,         // 0 = slow, 1 = fast
    depth=0.003,     // pitch modulation depth (sec)
    tremDepth=0.6,   // amplitude modulation depth
    mix=0.6,
	amp = 1
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
    Out.ar(out, XFade2.ar(input, wet, mix.linlin(0,1,-1,1))*amp);
}).add;

SynthDef(\mixer, {
    arg bus0, bus1, bus2, bus3, db0 = -6, db1 = -6, db2 = -6, db3 = -6, amp = 1;

    var sig0 = In.ar(bus0, 2) * db0.dbamp;
    var sig1 = In.ar(bus1, 2) * db1.dbamp;
	var sig2 = In.ar(bus2, 2) * db2.dbamp;
    var sig3 = In.ar(bus3, 2) * db3.dbamp;

    var mix = sig0 + sig1 + sig2 + sig3;

    mix = mix * 0.5 * amp;              // headroom
    mix = Limiter.ar(mix, 0.9);   // safety

    Out.ar(0, mix);
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
    |out=0, amp=0.7, toneFreq=180, noiseAmp=1, decay=0.25|
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



SynthDef(\doubleBass, {
    |out=0, freq=55, amp=0.6, gate=1,
     decay=100, coef=0.93, pluckAmp=1,
     pan=0|

    var env, excitation, string, body, fingerNoise;
    var signal, muteEnv, mutedSignal;

    // Main envelope (sustain controlled by gate)
    env = EnvGen.kr(
        Env.asr(0.01, 1, 0.4, curve: -2),
        gate,
        doneAction: 2
    );

    // Pluck excitation
    excitation = PinkNoise.ar(pluckAmp)
        * EnvGen.kr(Env.perc(0.001, 0.02));

    // String model
    string = Pluck.ar(
        in: excitation,
        trig: 1,
        maxdelaytime: 0.1,
        delaytime: freq.reciprocal,
        decaytime: decay,
        coef: coef
    );

    // Body resonance
    body =
        BPF.ar(string, 120, 0.5) * 0.6 +
        BPF.ar(string, 250, 0.4) * 0.4 +
        BPF.ar(string, 500, 0.3) * 0.3;

    // Finger noise
    fingerNoise = HPF.ar(WhiteNoise.ar(0.02), 2000)
        * EnvGen.kr(Env.perc(0.001, 0.03))*0.3;

    signal = string + body + fingerNoise;

    // Saturation
    signal = tanh(signal * 1.5);

    Out.ar(out, Pan2.ar(signal * env * amp, pan));
}).add;

SynthDef(\synthBass, {
    |out=0, freq=60, amp=0.3, gate=1,
     cutoff=100, res=0.3, envAmt=4,
     atk=0.005, dec=0.15, sus=0.6, rel=0.2,
     waveMix=0.7, drive=1.5, pan=0|

    var osc, sub, noise, filterEnv, ampEnv, signal;

    // --- Oscillators ---
    osc = Mix([
        Saw.ar(freq) * waveMix,
        Pulse.ar(freq, 0.5) * (1 - waveMix)
    ]);

    // Sub oscillator (adds weight)
    sub = SinOsc.ar(freq * 0.5) * 0.5;

    // Optional slight noise for grit
    noise = WhiteNoise.ar(0.02);

    // --- Filter Envelope (punch) ---
    filterEnv = EnvGen.kr(
        Env.perc(atk, dec, curve: -4)
    );

    // --- Amplitude Envelope (frees synth) ---
    ampEnv = EnvGen.kr(
        Env.adsr(atk, dec, sus, rel, curve: -4),
        gate,
        doneAction: 2
    );

    // --- Filter (classic lowpass) ---
    signal = osc + sub + noise;

    signal = RLPF.ar(
        signal,
        cutoff * (1 + (filterEnv * envAmt)), // dynamic sweep
        res
    );

    // --- Saturation (analog character) ---
    signal = tanh(signal * drive);

    // --- Output ---
    Out.ar(out, Pan2.ar(signal * ampEnv * amp, pan));
}).add;

)