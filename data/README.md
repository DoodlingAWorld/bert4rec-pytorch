# Data

MovieLens-1M is downloaded on demand and git-ignored. `download_ml1m` fetches it from
GroupLens, falling back to a Hugging Face mirror of `ratings.dat` when the primary host is
unreachable. Preprocessing (5-core filter, chronological ordering, leave-one-out split)
matches the SASRec reproduction so the two models are compared on identical data.
