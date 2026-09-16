# Data

The eye-tracking data are **not included** in this repository.

To run the pipeline, place the raw event-level file here:

```
data/raw/all\_data\_analysis\_komplett.csv
```

Alternatively, pass a different path:

```bash
python 00\_preprocess.py --mode all --raw-data path/to/file.csv
```

Preprocessed feature tables are written to `data/processed/<mode>/`.

## Expected columns

One row per eye-movement event (foveation or blink) with the following columns:

|Column|Description|
|-|-|
|`subject`, `trial\_Cntr`, `trial\_id`|Participant, trial counter, stimulus video|
|`task`|`freeviewing`, `search`, `memorization`, `balance\_search`, `balance\_memorization`|
|`ans\_cor`|Whether the trial was answered correctly|
|`gondola`, `num\_houses`, `train`|Number of these objects present in the video|
|`event`|`FOV` or `BLINK`|
|`duration\_ms`, `fov\_start`, `fov\_end`|Event duration and timing within the trial (ms)|
|`x\_start`, `x\_end`, `y\_start`, `y\_end`|Gaze positions|
|`fov\_velo\_abs\_mean`, `fov\_saliency\_mean`|Foveation velocity and saliency|
|`fov\_p\_median`, `fov\_p\_mean`, `fov\_p\_min`, `fov\_p\_max`|Pupil size during foveation|
|`sac\_velo\_abs\_mean`, `sac\_amp\_dva`, `sac\_angle\_h`, `sac\_angle\_p`|Saccade velocity, amplitude and angles|
|`gt\_object`, `gt\_object\_category`|Object looked at and its category|
|`fov\_category`|Foveation category (`B`: Background, `D`: Detection, `I`: Inspection, `R`: Return)|



