# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Dikac17/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Dikac17/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Hour-by-hour Twitch viewer forecast for Grand Theft Auto VI's launch day (19 Nov 2026): a calibrated k-nearest-launch quantile forecaster over 40+ past big game launches. Given the first hour's viewer count, it predicts each of hours 2–24 as five quantiles (5%, 25%, 50%, 75%, 95%).

Method: for each hour, take the 14 past launches whose hour-1 print is closest (in log space) to this launch's hour-1 print, and use the empirical quantiles of their log-ratios ln(viewers_h / viewers_1). Launch-day curves rise with the rolling midnight unlock and the size of the launch decides how steep that rise is — small categories can rise 100x from a near-empty first hour, while the biggest launches stay near 1x — so conditioning on size is what beats the persistence baseline. Local validation on the four practice launches: pinball loss 0.224 vs the house baseline's 0.387.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=1xzg4p72&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
