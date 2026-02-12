# Synch

A music recommendation engine that understands both sound and meaning, combining audio signals and semantics.

**Work in Progress**

## Current Status

| Component | Status |
|-----------|--------|
| Metadata extraction | **Complete** |
| Low-level audio features | **Complete** |
| High-level audio features | **In Progress** |
| Lyric embeddings | **Planned** |
| KNN similarity search | **Planned** |
| GUI | **Planned** |

**Dataset:** Currently 5,000+ tracks, scaled to be 10,000+.

**Audio Analysis**
- Feature extraction via [Essentia](https://essentia.upf.edu/)

**Lyrics Analysis**
- Semantic embeddings generated with **Sentence-BERT**

**Similarity Search**
- **K-Nearest Neighbors** algorithm

## Data Sources

- Audio files from undisclosed source
- Metadata verified via [MusicBrainz](https://musicbrainz.org/)
- Lyrics from [Genius](https://genius.com/)

**Note:** This project runs on **WSL** due to limitations Windows exhibits.
