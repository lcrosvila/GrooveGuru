# GrooveGuru

Welcome to the messy world of generating StepMania charts using (for the moment) Transformers! 

## Previous version (version 7):
Get the data from [DDR Classics](https://search.stepmaniaonline.net/pack/id/404)

Adapt the file `check_songs.py` to where you put the folder.

Then run `attempt7.py`

(it's a bit messy and requires tweaking, but it should kind of work)
(also, it doesn't technically generate the whole .sm file, but rather it just gives you the steps)

## Current version (version 8):
You can download the data with 
`download_tracks.sh`

Then parse the charts by running:
```
python smdataset.py dataset dataset/parsed_files
```

When checking the time increments between steps, we can see that for most tracks the minimum step is of 3ms (except for 8 tracks). If we remove those from the dataset we could pad the charts to have a step of 3ms, and use a sliding window of 3ms to preprocess the audio.

(TO DO) Train a new model with the new dataset, as now we have a better alignment between steps and time steps. The generated json files look something like this:

<sub>
  ... "charts": [{"type": "dance-single", "desc_or_author": "K. Ward", "difficulty_coarse": "Easy", "difficulty_fine": 4, "notes": [[0.002, "0000"], [0.3890967741935484, "0000"], [0.7761935483870968, "0000"], [1.163290322580645, "0000"], [1.5503870967741935, "0000"], [1.937483870967742, "0000"], ...
</sub>

Which is super useful! I'm sure we can figure out a way to exploit that. But we'll have to rethink the model's architecture completely.
