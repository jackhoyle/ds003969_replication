# EEG Replication Project

Aim: build a simple EEG processing pipeline then replicate some basic results from a paper (https://pmc.ncbi.nlm.nih.gov/articles/PMC5261734/)

Data is at: https://openneuro.org/datasets/ds003969/versions/1.0.0


##  Steps

1. Install relevant packages 
2. Explore data in a notebook 
3. Get familiar with analysis tools like MNE-python, MNE-BIDS
4. Produce some standardised analysis and explore data formats
5. Implement independently some analytical functions to produce desired outputs
6. Compare results to package implementations for similarity

## Curiosities for later exploration

* Authors use right mastoid as reference, why not left, do these give similar results?

* Authors don't seem clear on what hypotheses they tested? it's probably worth thinking about how many hypotheses they likely tested to end up with their final results and adjusting things accordingly.

* Authors make some conclusions regarding frequency bands including 100Hz but they don't do much to deal with 50Hz line noise. They exclude this 50Hz band but they don't account for harmonics. Is this an issue? Manual inspections doesn't show much going on at 100Hz that is abnormal.

## Replication summary

* Referenced to right mastoid (EXG6)
* Subsampled to 256Hz
* Bandpass filtered with same filter
* Implented the `pop_rejcont` function in python, manual  inspection of it's flagging seems sensible
* The next step involved manual inspection then rerunning `pop_rejcont` with lower tolerance over certain series. Given they don't specify how they chose these I can't replicate it and shall hope it's unimportant.


### 30/9/26

I've manage to figure out how to load data in this standardised format using `mne-bids`, then had a play around with analysing it using `mne`.

The data is quite simple to use but I'm running into some annoying OOM issues as well as having too slow internet.

Next steps are to get more familiar with `mne` and what it can do, then try to step-by-step replicate the authors' pipeline. Initially let's try do this exactly whilst commenting on anything that seems off, we can revisit robustness to those later.

A useful note on the electrodes from the data website:

```Hello, Here are the details of the positions of the external electrodes:
Ext1 --> left eye corner
Ext 2 --> right eye corner
Ext 3 --> left eye eyebrow (above)
Ext 4 --> left eye below
Ext 7 --> middle of the collar bone
Ext 5 --> left mastoide
Ext 6 --> right mastoide
EXT 8 was set up with an extra electrode for Fp1, due to a defect of that electrode on the Biosemi electrode set.
So from that you have: 
 Ext1 and Ext2: HEOG; 
 Ext3 and Ext 4: VEOG (blinks); 
Ext 7: ECG
Ext5: M1
Ext6: M2
Ext8: FP1

Hope it helps!
```

Right mastoid is used for reference by authors. Setting this removes most of the obvious line artifacts.

Implementing pop_rejcont in python involved getting very much into the weeds. Lots of details left out of the paper and needed to be guessed at from the matlab defaults, claude handled implementation and wrote tests.

Have made it up to manual bad channel inspection, I have a plan for this tomorrow.

### 1/10/26

Day two and I'll start properly later but a nice side quest has presented itself.

I want ideally a pipeline I fully understand and trust, that I can point at new data sources and be happy with with little manual work. Hence, I want to automate my bad channel detection. So plan is to build a simple bad channel detection widget, run this, then still do the full manual detection to confirm it's working after. I had the idea in my mind to look for some sauna EEG data and it would be nice to just run and verify rather than implement fresh each time. Given all this it's worth pulling all this out into my own convenience package in the future.

I've added a simple minimum variance threshold in the automated pipeline now, things are running well.

Next steps:

* Make notebook for going through manual annotation
* Come up with a format for storing processed data and record of bad channels
* Figure out what's going on with referencing interacting badly with `pop_rejcont` and filtering




Everything looks fine with referencing, it's doing the simple subtraction transform one expects.

Looking more closeltt at the data it's clear that referencing messes up the broken channels, with the broken channels just becoming the inverse of the reference channel, then looking like normal data later on once processed. We definitely need to identify bad channels by processing but not referencing then inspecting.

Still need to play around with this more.

### 2/10/26

Planned pipeline:

* Preprocess without referencing to identify bad channels
* Save bad channels
* Do preprocessing with referencing
* Then, infomax ICA on processed data
* Save processsed data
 
for the ICA part they reference this paper https://pubmed.ncbi.nlm.nih.gov/17188898/

It seems they use an automated implementation in Matlab. MNE/python also has this functionality and I'll try to use something similar.

I also stumbled upon the `autoreject` package for bad channel/epoch detection, worth looking at this in more detail for the future rather than redoing the janky stuff in this paper.

I figured out why Fp1 is bad: it isn't used and EXG8 is used instead, just need to swap the data in there. This makes things look more sane. pop_rejcont seems to be working as intendd.

OK so new plan now that referencing and other problems resolved:

* Do preprocessing
* Suggest bad channels, visually inspect to confirm
* Do ICA
* Save all processed data 

For now I have the bad channel suggestion automated using the built in `mne` function based on waveform distance, but also I built my own that takes that approach but applies it to spectral distance over 1-45Hz. These seem complementary and so far the suggestions seem good. I'll run through a bunch before the final run through to calibrate the thresholf I want to use here.

Now I need to look into ICA. The authors are not overly specific so it sound like they use a vanilla package version. I shall do similarly, using the python version but making the settings close where possible.

It seems there is quite an art to detecting ICA components. https://labeling.ucsd.edu/tutorial/labels is a very good resource for this. I'm not quite confident yet to go through each manually and call out which they are. Luckily MNE has automatic detection functions in place for muscle and EOG artefacts which seem to behave sensibly on some test cases. I'm going to just use these.