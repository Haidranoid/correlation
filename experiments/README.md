# Experiments

**TL;DR:** 71 attempts to predict US stock moves, from correlations between stocks, outside data, machine learning and fourteen published strategies. None makes money reliably after trading costs. Each file is named by its result after costs: points a year above simply holding SPY, or a market-neutral book's own net return. The one open lead, the monthly top-10 picks, needs data that includes delisted companies. What works is risk: how stocks move together on the same day.

## How to read the file names

Each file is named `[after costs]-[description].md`. The number is the result after trading costs of 5 bps a side, averaged over the file's test years:
- **Long-only strategies** (the judge's 20-stock accounts, stockai's top-10 accounts, portfolios, timing rules): the strategy's yearly return minus SPY's total return. `-5.2` means it made 5.2 points a year less than simply holding SPY.
- **Market-neutral long-short books** (pairs, hedged books, overlays): the book's own net yearly return, since it is already market-neutral.
- **No trading:** five files test something that isn't a trading strategy: the census, the lag spectrum, the forecast's accuracy, the outside-data leaders and the risk tools. They keep their own statistic, and the table says "no trading". Each names the file with its money test.
- **`untested`:** anything not yet measured. There is none at the moment.
  - Brandon's gravity file (run 2026-10-24) and the five published effects run on 2026-10-25 were named `untested-<topic>.md` until then. Some of their Rules sections, scripts and Decisions still use those names: the Rules are frozen, and the scripts' sha256 hashes are recorded for the forward tests.

**Reading the numbers:**
- **A positive number after costs isn't a success by itself.** Check the file's own tests, which the Verdict column summarizes. The positives here mostly rest on one of these, and most aren't statistically significant:
  - crisis years: overnight auctions is +15.7 almost entirely from 2008, 2009 and 2020;
  - statistics revised after the fact: outside data at one week;
  - survivor-only data: the price data holds only companies still listed, which flatters any long-only stock account against SPY, most of all the monthly top-10 picks.
- **5 bps a side is realistic for liquid stocks.** Many files also report 20 bps, where most of the positive numbers turn negative. The next-day LightGBM is +5.8 here and loses 35.7% a year at 20 bps.
- **Special cases:**
  - the twin gap fade uses its tradable version, entered at 9:35 over the 58 days of intraday data, because the opening-price book it measured can't be traded;
  - intraday lead-lag trades every five minutes and loses everything (−100);
  - the 16-month out-of-sample test is scaled to a year.
- **Leading minus signs:** in the shell, a file whose name starts with a minus needs `./` in front of it, for example `less ./-5.2-peer-momentum.md`.

**The "Before costs, vs random" column** is the number these files carried before the switch: the change over random before trading costs. What it measures:
- **Money tests:** percentage points of yearly return above random, averaged over the test years. `+3.0` means the picks made 3% a year more than random.
  - In correlation2's tests, "random" is the median of 1,000 random accounts that hold as many stocks and trade as often.
  - In stockai's top-10 tests, it is the average eligible stock, which is what random picks earn on average.
- **Tests without a money step:**
  - for the census tests and intraday lead-lag, the percent more strong relationships than chance produces (`+2.0` = 1.02× chance);
  - for the forecast, points of direction accuracy above the base rate;
  - for the risk tools, the percent lower next-year volatility than a random choice;
  - for free timing, the extra yearly return from timing trades the portfolio makes anyway, against making the same trades on a random day of the window;
  - for analog days, the judge's account on stocks mapped to the forecast sectors. It is a proxy; the real test is the sector-fund long-short in the file;
  - for market timing, the yearly return above random timings that keep the rule's own exposure. The variance risk premium and credit-spread rules use the same comparison;
  - for excess growth, the rebalanced portfolio's yearly return minus buy-and-hold of the same stocks, both before costs;
  - for the twin gap fade, 252 × the daily return of its own $1-long/$1-short book, minus the median of 1,000 random books with the same structure and costs. It is a daily long-short spread, not a 20-stock account, so it is far larger than the other numbers and can't be compared with them;
  - for overnight auctions, a 20-stock account held only overnight (bought in the closing auction, sold at the next open) against 1,000 random 20-stock overnight accounts. Volatile stocks' overnight moves are large, so it isn't comparable with the judge's close-to-close accounts;
  - for correlation portfolios, (Sharpe ratio of minimum variance − Sharpe ratio of inverse volatility) × 15: points of yearly return at 15% volatility, with no random accounts;
  - for risk-managed momentum, the same Sharpe difference × 15, against plain momentum;
  - for trend following, points of yearly return above the median of 1,000 portfolios of the same funds with each fund-month's long or short drawn at random;
  - for pairs trading, the yearly return on committed capital minus the median of 1,000 books of random pairs (Gatev et al.'s own comparison);
  - for the turn of the month, the yearly return above random timings that hold SPY the same number of days a month.
- **Shorter windows:** a few judge numbers cover only the years with data (lazy prices and hidden text peers, 2007–2022) or the years after the selection period (formula mining, 2015–2025).
- **Beating random isn't making money.** The random accounts trade as often as the strategy and pay the same costs, so a strategy can beat them and still lose money. That's why the file names now use the after-cost number.

## What has been tried (in the order it was done)

| After costs | Before costs, vs random | Experiment | Project | Verdict |
|---|---|---|---|---|
| −3.6 | −1.6 | [The original Rails app's pair finder](-3.6-v1-pair-finder.md) (built 2025; tested out of sample 2026-10-21) | correlation v1 | Finds what chance finds; no edge |
| −22.0 | +6.9 | [v1's idea on a holdout year](-22.0-pair-lead-lag-holdout.md) (one year, p = 0.245) | correlation2 (Python) | No edge |
| no trading | +0.0 | [Does lagged structure repeat between periods?](+0.0-lag-spectrum.md) | correlation2 (Python) | No; same-day structure does |
| no trading | +2.0 | [The census: every pair, offset and move length](+2.0-correlation-census.md) | correlation2 | Lagged pairs at chance |
| −11.4 | +3.0 | [Trading the census's pairs, 19 years](-11.4-census-picks.md) (not significant) | correlation2 | No edge |
| +5.4 | +11.8 | [Positive control: 5-day reversal](+5.4-reversal-control.md) | correlation2 | Detected; fading since 2021 |
| −5.2 | −2.1 | [Peer momentum (co-movement peers)](-5.2-peer-momentum.md) | correlation2 | No edge (slightly reversed) |
| no trading | +0.8 | [Forecasting a stock from its relationships](+0.8-forecast-from-relationships.md) (hit rate) | correlation2 | No practical skill |
| no trading | +4.0 | [1,753 outside series as leaders](+4.0-outside-data-leaders.md) (noise also +4.0) | correlation2 | Nothing leads |
| −19.8 | +3.1 | [stockGPT, next day](-19.8-stockgpt-next-day.md) | stockai | Real gross signal; no edge |
| −4.9 | +14.4 | [LightGBM on correlation2's daily move](-4.9-lightgbm-next-day.md) | stockai | No real edge; exposed a census artifact |
| +5.8 | +35.8 | [LightGBM on actual next-day returns](+5.8-lightgbm-next-day-actual-returns.md) | stockai | Real but gone in a day; loses money after costs |
| +2.1 | +4.5 | [12-1 momentum](+2.1-momentum-12-1.md) | stockai | Matches SPY after costs; no edge |
| +18.6 | +26.2 | [Top 10 for the next month](+18.6-monthly-top10-picks.md) | stockai | Lead, but survivorship-flattered; a stress test without new data says it survives unless its picks vanish about 2.4 times as often as similar stocks |
| +7.4 | +16.0 | [Top 10 for the next week](+7.4-weekly-top10-picks.md) | stockai | Small, fairly robust; costs exceed it |
| +3.6 | +5.5 | [Top 10 for the next two months](+3.6-two-month-top10-picks.md) (t = 0.7) | stockai | No edge |
| −5.7 | +1.5 | [stockGPT, top 10 for the next week](-5.7-stockgpt-1-week.md) | stockai | No edge |
| −7.2 | −4.2 | [stockGPT, top 10 for the next month](-7.2-stockgpt-1-month.md) | stockai | No edge |
| −3.0 | −0.4 | [stockGPT, top 10 for the next two months](-3.0-stockgpt-2-months.md) | stockai | No edge |
| +20.9 | +23.0 | [Outside data as model inputs, one week](+20.9-outside-data-inputs-1-week.md) | stockai | Gain rested on hindsight-revised data |
| +10.9 | +12.4 | [Outside data as model inputs, one month](+10.9-outside-data-inputs-1-month.md) | stockai | Worse than stock only (+26.2) |
| −6.6 | −0.2 | [Outside data as model inputs, two months](-6.6-outside-data-inputs-2-months.md) | stockai | Worse than stock only (+5.5) |
| −19.9 | +0.6 | [The 16 months the census never saw](-19.9-out-of-sample-16-months.md) | correlation2 | No edge |
| −4.5 | +5.6 | [Groups that move first (Bennett et al. 2022)](-4.5-cluster-lead-lag.md) (+6.2 on all exchanges) | correlation2 | Real gross (slow followers); no money after costs |
| no trading | +5.1 | [Same-day co-movement tools](+5.1-co-movement-tools.md) (% lower next-year volatility; built 2026-10-20, tested 2026-10-21) | correlation2 | Works: the advice beats random choices out of sample (risk, not returns) |
| −7.5 | +3.3 | [Bystanders](-7.5-bystanders.md) (not significant; long-short −0.9 bps/day) | correlation2 (Python) | No give-back; co-movers follow the news instead |
| −1.9 | +0.1 | [Unequal move lengths: does A's week predict B's month?](-1.9-unequal-move-lengths.md) (mean of 20 combinations) | correlation2 (Python) | No combination predicts; week → month +2.6, p = 0.13 |
| −9.1 | −0.1 | [Phantom portfolios](-9.1-phantom-portfolios.md) | correlation2 (Python) | Unwinds keep falling instead of bouncing |
| +0.1 | +1.4 | [Imposters and undiscovered members](+0.1-imposters.md) (not significant) | correlation2 (Python) | The gap closes, but prices don't correct |
| +0.7 | +0.7 | [Free timing: signals used only on trades you make anyway](+0.7-free-timing.md) (vs a random execution day; t = 0.9) | correlation2 (Python) | About 5 bps a trade, not significant |
| −5.5 | −4.4 | [Betting against correlation](-5.5-betting-against-correlation.md) (published effect; low-beta picks trail in rising markets) | correlation2 (Python) | Alpha +3% a year, t = 1.3; nothing beyond betting against beta and low volatility |
| +1.1 | +3.3 | [Twin reversal](+1.1-twin-reversal.md) (published effect's variant; not significant) | correlation2 (Python) | The hedged long-short loses; plain reversal earns nothing in large caps either |
| −7.7 | +0.8 | [Volume clock: buy before the news you can predict](-7.7-volume-clock.md) (published premium, new calendar) | correlation2 (Python) | The clock hits 30% of windows (needs 40%); the premium is −19 bps an event |
| −11.5 | −6.6 | [Analog days: has the market been here before?](-11.5-analog-days.md) (known method, new application; stock proxy) | correlation2 (Python) | Sector-fund long-short +0.2 bps a week after costs, t 0.02 |
| −8.2 | +0.2 | [Silent-peer news drift](-8.2-silent-peer-drift.md) | correlation2 (Python) | Silent minus loud +0.06%, t 0.2; no news drift at all to sort |
| −6.0 | +10.3 | [Night and day: does the catch-up come overnight or during the day?](-6.0-night-and-day.md) (the judge trades close to close, so it includes the overnight leg) | correlation2 (Python) | Real catch-up, but the intraday book nets +3.6 bps a day, t 2.4 (needs 3); lost money in the unseen months |
| −16.2 | +148.8 | [Twin gap fade](-16.2-twin-gap-fade.md) (252 × a daily long-short spread; not comparable with the other numbers) | correlation2 (Python) | Passes every test on daily data (t 29), but the profit lives in the opening print: entry at 9:35 keeps 0.22 of it, −10 bps a day after costs |
| +2.5 | +8.2 | [Close location: where in its range a stock closes](+2.5-close-location.md) (the number is 5-day reversal's) | correlation2 (Python) | Nothing beyond reversal: controlled t −1.4 |
| −1.3 | −2.2 | [Lazy prices: companies that rewrite their 10-K](-1.3-lazy-prices.md) (published effect; years with text, 2007–2022) | correlation2 (Python) | Long-short +0.03% a month net, t 0.26 |
| −3.2 | −0.3 | [Market timing from co-movement](-3.2-market-timing.md) (published effect; against random timing at the same exposure) | correlation2 (Python) | Same as random timing; volatility-managed SPY did better |
| −3.2 | +2.7 | [Signal ensemble](-3.2-signal-ensemble.md) (standard practice) | correlation2 (Python) | IC t 3.0, but the buffered long-short nets +2.4 bps a week, t 0.27 |
| −4.3 | −2.9 | [Loners: stocks that move with nothing](-4.3-loners.md) | correlation2 (Python) | +0.26% a month net, t 1.5 (needs 2.5); lost 23% in the unseen months |
| −0.1 | +1.4 | [Twin momentum](-0.1-twin-momentum.md) (published effect's variant) | correlation2 (Python) | The twin hedge cuts risk by 40%; the signal adds nothing over plain momentum |
| −2.4 | +6.0 | [Formula mining with an honest holdout](-2.4-formula-mining.md) (test years 2015–2025 only) | correlation2 (Python) | All 20 picks were reversal variants; holdout IC t 1.4 |
| +2.6 | +7.6 | [Group breadth thrust](+2.6-breadth-thrust.md) | correlation2 (Python) | +15.5 bps a month after thrusts, t 1.3 (needs +50 at t 3) |
| +15.7 | +54.1 | [Overnight momentum at the auctions](+15.7-overnight-auctions.md) (published effect's variant; a 20-stock overnight-only account, not comparable) | correlation2 (Python) | The hedged book nets +1 bp a day, t 2.1 (needs 3), and loses in large caps, after 2020 and in the unseen months; the profit sits in the opening print |
| −1.6 | −1.4 | [Overnight peers](-1.6-overnight-peers.md) | correlation2 (Python) | Real economic peers (Apple's are its suppliers), but −0.31% a month, t −0.9 |
| +4.6 | +4.7 | [Hidden text peers](+4.6-hidden-text-peers.md) (published effect's variant; years with text, 2007–2022) | correlation2 (Python) | −0.11% a month net, t −0.33; hidden peers no better than visible |
| −3.8 | −2.0 | [Statistical-factor momentum](-3.8-factor-momentum.md) | correlation2 (Python) | +0.07% a month net, t 0.5; mostly plain momentum |
| −4.6 | −0.4 | [Correlation-built portfolios](-4.6-correlation-portfolios.md) (published methods; Sharpe difference × 15 against inverse volatility) | correlation2 (Python) | A quarter to a third less risk in every period, unseen months included, but no better Sharpe |
| −100.0 | +23.6 | [Intraday lead-lag in 2026](-100.0-intraday-lead-lag.md) (% more lagged pairs than chance, 5-minute bars) | correlation2 (Python) | Needs +100%. The 1-minute lead-lag is nonsynchronous trading and breaks even at 0.45 bp a side |
| +0.0 | +6.0 | [Off-exchange short volume against peers](+0.0-short-volume.md) (FINRA's daily files; sign learned on 2010–2015) | correlation2 (Python) | Ranks a little (IC t 2.9 in 2016–2019; t 1.1 once lagged a day to when FINRA posts), but the weekly book nets −0.5 bps a week |
| −3.5 | −1.5 | [Theme joiners](-3.5-theme-joiners.md) | correlation2 (Python) | Wrong sign: −17 bps a month after the controls, t −0.65 |
| −10.2 | +0.3 | [Decoupling from the twin](-10.2-decoupling.md) | correlation2 (Python) | The drift reverses slightly (−82 bps, t −0.8); the events are mostly market-wide correlation shifts |
| −0.6 | +7.7 | [Announcement-day beta](-0.6-announcement-beta.md) (published effect, new to this data; the strategy itself was published in 2023) | correlation2 (Python) | Replicates in 2006–2019 (slope gap t 3.1), but +9.5 bps per announcement day after costs, t 1.0; gone since 2020 |
| +1.2 | +1.4 | [Trend following across asset-class funds](+1.2-trend-following.md) (published, Moskowitz, Ooi & Pedersen 2012; against random-sign portfolios) | correlation2 (Python) | Matches AQR's own series (Sharpe within 0.02); gained in 2008 and 2022, but +1.8% a year, t 1.5 over 2008–2019 |
| −0.2 | +1.9 | [Return seasonality](-0.2-seasonality.md) (published, Heston & Sadka 2008) | correlation2 (Python) | −0.23% a month net, t −1.05; neither half of the published pattern shows |
| −4.4 | −3.8 | [Betting against beta](-4.4-betting-against-beta.md) (published, Frazzini & Pedersen 2014; the judge buys the low-beta half) | correlation2 (Python) | Tracks AQR's series (r 0.67), but +0.25% a month, t 0.9 on liquid stocks; losing since 2020 |
| +3.3 | +5.4 | [Risk-managed momentum](+3.3-risk-managed-momentum.md) (published, Barroso & Santa-Clara 2015; Sharpe difference × 15 against plain momentum) | correlation2 (Python) | The scaling works (Sharpe −0.26 → +0.21, p 0.001; 2009 crash −78% → −22%), but the book earns +0.20% a month, t 0.66, because momentum itself earns nothing here |
| −7.4 | −5.3 | [The 52-week high](-7.4-52-week-high.md) (published, George & Hwang 2004) | correlation2 (Python) | −0.38% a period net, t −0.97; it is momentum here (R² 0.75) and crashed in 2009; replications since publication agree |
| +0.8 | +0.3 | [Pairs trading, distance method](+0.8-pairs-trading.md) (published, Gatev, Goetzmann & Rouwenhorst 2006; against random pairs) | correlation2 (Python) | Behaves like the paper's rule, but earns nothing over 2006–2019 (t 0.04), below random pairs; a small revival in turbulent stretches since 2020 |
| −6.8 | +7.8 | [Statistical arbitrage against funds](-6.8-etf-stat-arb.md) (published, Avellaneda & Lee 2010) | correlation2 (Python) | Real before costs (Sharpe 1.24), t 0.8 after; negative when filled at the next close |
| −3.6 | −1.7 | [Lottery stocks (MAX)](-3.6-max-lottery.md) (published, Bali, Cakici & Whitelaw 2011) | correlation2 (Python) | Wrong sign: −0.49% a period net, t −0.9, and the controlled coefficient positive; Chen & Zimmermann's survivor-free version also faded to +0.2% since publication |
| −9.6 | +0.1 | [Turn of the month](-9.6-turn-of-month.md) (published, McConnell & Xu 2008; against random timings) | correlation2 (Python) | Reproduces 1926–2005 exactly (t 8.2), then nothing in 2006–2025 (t −0.07) |
| +2.0 | +3.5 | [Frog in the pan](+2.0-frog-in-the-pan.md) (published, Da, Gurun & Warachka 2014) | correlation2 (Python) | −0.71% a month net, t −1.05, and continuous-information momentum did *worse* than discrete (t −1.30); the paper's six-month spread was +8.0% |
| −5.6 | −0.4 | [Cost-aware trading on the weekly top 10](-5.6-cost-aware-trading.md) (published methods; follow-up to `+7.4-weekly-top10-picks.md`) | stockai + correlation2 (Python) | The weekly signal lasts one week (weeks 2–4 IC t ≤ 1.1). The chosen rule cut trading by two thirds but turned $10,000 into $10,520 against SPY's $25,375 at 20 bps (p 0.54 against random) |
| −16.7 | +1.8 | [Gravity: correlated stocks as attractors in price, time and both](-16.7-brandon-gravity.md) (Brandon's idea; mean of 30 force laws, the judge's 20-stock accounts) | correlation2 (Python) | All six tests fail Holm (best t 1.85). The measured pull grows with distance, like pairs trading's spring, and random peers pull as hard; the time kernel has no peak at the offset |
| −2.2 | +1.6 | [The variance risk premium](-2.2-variance-risk-premium.md) (published, Bollerslev, Tauchen & Zhou 2009; the premium's tilt on variance timing, against random reorderings of its own exposures) | correlation2 (Python) | Clark-West t 1.43 (needed 2). Its R² of +5.3% comes from 2008–2009 alone, and 2020–2025 was −28%. The tilt's Sharpe gain over variance timing is one trade (p 0.24) |
| −4.7 | +0.0 | [Credit-spread timing](-4.7-credit-spread-timing.md) (the Baa spread's weekly change; a SPY-or-cash rule against random reorderings of its own exposures) | correlation2 (Python) | Wrong sign: a widening spread came before slightly higher returns, and forecasts got 1.7% worse (Clark-West t −1.52). Stocks lead the spread (t −4.3), not the reverse |
| +0.3 | +0.5 | [Excess growth: rebalancing against buy-and-hold](+0.3-excess-growth.md) (published, Fernholz 2002; rebalanced minus buy-and-hold of the same stocks, both before costs) | correlation2 (Python) | Daily rebalancing +0.58 a year after costs (t 1.13), all from 2008–2009. γ* predicted the extra growth, but buy-and-hold collected 80% of it too. Survivor-free sector funds and a century of industries show no premium |
| +3.7 | +5.4 | [Amihud illiquidity](+3.7-amihud-illiquidity.md) (published, Amihud 2002; the 20 least liquid stocks of the judge's liquid set) | correlation2 (Python) | Dollar volume and volatility are 98–99% of the measure; what is left has t 0.42 in the liquid set and 1.34 with no volume floor. The account's margin is 2008–09 and 2020–21 alone; its median year lost 2.2 |
| −23.4 | −0.9 | [Volume–return dynamics](-23.4-volume-return.md) (published, Llorente, Michaely, Saar & Wang 2002; the judge's account traded daily, a day late) | correlation2 (Python) | Both primaries fail (t 0.72 and −1.49). Daily turnover costs over 20 points a year; weekly, −3.7. The pattern is real across stocks for the next day (t −6.9) but gone a day later, when a trader can act |

## Lessons from all of it

- **Same-day co-movement is the useful part.** It persists, and the tools built on it beat random choices the following year: 5.1% lower volatility from the diversifier pick, and 68% of a stock's risk left after the top hedge against 91% for a random fund.
- **Lagged correlation is chance at every combination of move lengths tried,** once the chance comparison is fair. The unequal-move-lengths test's block sign-flip on residuals fixes the old null's false "excess" at long moves (1.87× on pure noise, now at most 1.04×).
- **News spreads, it doesn't splash.** Co-movers that move on another company's news continue when that company's move continues and reverse when it reverses (bystanders). The move carries information, not mechanical drag.
- **Coordinated basket sell-offs keep going.** Baskets that fall together mostly do so on big market days, and keep falling (phantom portfolios).
- **The market does reclassify companies,** toward what their 10-K says, over about a year. But the price doesn't give it away ahead of time (imposters).
- **Short-term reversal is gone in large caps.** Among the 500 most-traded stocks since 2006, plain 1-month reversal earns +0.12% a month (t 0.2), and residuals against a stock's tracking fund do no better (twin reversal). The reversal the harness detects lives in smaller stocks.
- **Even free use of the short-term signals adds little.** Timing trades a portfolio makes anyway gains about 5 bps a trade at the next open, not significant (free timing).
- **The calendar effects are gone too.** The announcement premium is negative over 2006–2019 here (volume clock), consistent with reports that it disappeared after the 2004 8-K rule.
- **News days don't drift.** After a stock's own volume-spike news, the next month's drift is −0.05% (t −0.6), whatever its peers did (silent-peer drift). The echo does classify news, since loud days have twice as many co-movers trading heavy volume, but the price treats both kinds alike.
- **The catch-up to the market is real, but it now happens overnight.** Slow followers do catch up on yesterday's market move. Since 2020 the catch-up has moved into the next opening gap, out of a daily trader's reach (night and day). The cluster lead-lag near-miss splits the same way: its intraday part made +2.7 bps a day (t 2.0) over 2008–2019, and has been negative since 2020.
- **Combining weak signals doesn't beat costs.** The ensemble ranked stocks better than any one signal, but a long-short earns at the extremes, and averaging dilutes reversal's extremes (signal ensemble).
- **Brute force rediscovers reversal.** Of 1,352 price and volume formulas, the 20 best on 2005–2014 were all reversal variants, and they had faded by 2015–2025 (formula mining).
- **A huge daily-bar result can live in one print.** The twin gap fade made +74 bps a day from the official open and kept a fifth of it five minutes later. Anything that enters at the open or the close needs intraday bars before it counts, and Yahoo's free 1- and 5-minute bars cover the last 30 and 60 days.
- **More published effects don't show here:**
  - lazy prices: t 0.26 over 2007–2021;
  - residual momentum against the twin: no better than plain momentum, which itself earned nothing among the 500 largest;
  - co-movement market timing: no better than random timing.
- **Correlations build calmer portfolios, not better ones.** Minimum variance on a shrunk covariance had 63–76% of inverse volatility's risk in every period, unseen months included, and a 2008 drawdown of −36% against −52%. Its Sharpe ratio was no better (correlation portfolios). It is the same lesson as the co-movement tools: correlations predict risk.
- **The opening print, again.** Overnight drift collected through the auctions makes 11.6 bps a day before costs, and prices are 20 bps lower by 9:35 (overnight auctions). An opening-auction order does get that price, so this one is real but too small after costs.
- **Better peers don't rescue peer momentum.** Overnight co-movement finds real economic peers (Apple's suppliers), and 10-K text finds sensible ones (JPM → USB, NTRS), but neither predicts, and joining a hot group predicts slightly *lower* returns.
- **Some published effects replicate here before 2020 and not after:** the beta premium on announcement days (t 3.1 in 2006–2019, reversed in the unseen months), and FINRA's short volume (IC t 2.9 in 2016–2019, t 1.3 since).
- **FINRA posts each day's short-volume file after the close,** so a signal that uses day t's file can't trade at day t's close. Lagged a day, the short-volume IC fell from t 2.9 to t 1.1.
- **Published strategies decay, and the cleanest replications show it most clearly.** The turn-of-the-month code reproduces the 1926–2005 result (t 8.2) and finds nothing since (t −0.07). Trend following matches AQR's live series, whose Sharpe fell from 1.4 to 0.4 after publication. Seasonality, the 52-week high, MAX, pairs trading and betting against beta are flat or negative here after their papers, and independent replications (Chen & Zimmermann's open-source data) agree.
- **The parts of published strategies that survive are risk management.** Scaling momentum by its own volatility raised its Sharpe in every period and cut the 2009 crash from −78% to −22%; trend following gained in 2008 and 2022. Neither adds return on average.
- **Filling at the signal's own close flatters daily reversion.** A third of the fund-residual stat arb's gross profit comes from trading at the close that produced the signal. Filled a day later, it loses.
- **Betting against beta's live record comes from small stocks.** AQR's series still earns since publication (+0.50% a month, t 2.6), but the same construction on the liquid stocks here doesn't, as Novy-Marx & Velikov argue.
- **More data problems to know about:**
  - Since late 2019 the two price stores disagree on about a dozen ex-dividend days for some sector funds (XLB, XLE, XLK, XLY), by up to 1% a day.
  - The fund-like flag misses some funds filed as stocks (JEPQ, QQQE, ARTY, GARP), where no more-traded twin moves at 0.97 with them.
  - The new store (through 2026-10-19) has broken volumes around 2026-03-03: a few dozen shares for about 700 stocks.
  - The store is back-adjusted for dividends through about Oct–Nov 2022 for most payers, and raw after that, so overnight returns on later ex-dates are distorted.
- **Published low-risk effects mostly overlap.** On this data, betting against correlation is 1.33 × betting against beta − 1.02 × low volatility (R² 0.84), with nothing of its own left.
- **Turnover-matched random accounts are essential.** Without them the census "beat random" at p = 0.000, only because it traded less.
- **Volatility fools simple yardsticks.** Volatile picks win in good years, and calm picks earn steady small gains. Matched random accounts and comparable-stock yardsticks are needed.
- **Grade in actual returns.** correlation2's census variable contains a part of tomorrow's value that is known today (`-4.9-lightgbm-next-day.md`).
- **The data has survivors only.** Stocks delisted before 2025 are missing, which flatters anything buying recent losers (`+18.6-monthly-top10-picks.md`). The fix is Sharadar ($39 for a month); the code is ready in `stockai/docs/sharadar.md`.
- **Two look-aheads remain in correlation2's data:** the $5 price rule uses split-adjusted prices, and the bad-print repair looks one day ahead. Both are small, and both are still open.
- **One year of results is noise.** The same rule on two census grids gave +23.3% and +10.8% for the same year.

## How new experiments are graded

- **The shared grader:** `research/lib/c2eval.py` wraps stockai's judge, which is correlation2's $10,000 harness, bit-identical.
  - Write a scores array (stocks × days, NaN = no score) and grade it with `c2eval.judge`.
  - `c2eval.change_over_random` gives the number for the file name. 5-day reversal through it gives +11.8, matching `+5.4-reversal-control.md`.
- **Exports:** the old one (prices through 2025-06-27) is for the walk-forward over 2006–2025. The new one (through 2026-10-19, in `~/.local/share/c2-sandbox-2026-10-19/`) is for the 16 unseen months, used once at the end.
- **Fixed rules:** thresholds are set before results are seen. Decisions made while building are recorded in the file, marked as made before or after seeing results.
- **Baseline:** beat the nearest known effect, not just random. For example, a reversal-like effect must beat plain reversal.
- **Market-neutral results**, because the data holds only survivors. The judge's long-only account supplies the file-name number; the verdict rests on the file's own test plus the long-short.
- **Kill criteria in order.** The money test still runs so every file gets a number.
- **Published strategies (the fourth round)** pass at t ≥ 2 rather than 3, because a published result is the prior. Nearly all of their test years fall after the papers' samples, so these are out-of-sample tests in the McLean–Pontiff sense. Where a published series exists (AQR's BAB and TSMOM, French's UMD), our version's correlation with it was checked before any test was read.
- **Many tests at once:** ten tests at the usual bar give about one false pass by luck, so a pass counts only if the unseen months agree too. A pass far beyond any luck, like the twin gap fade's t of 29, raises a different question: whether the data is right.
- **Wording:** the project's terms (correlation period, move length, offset, A moves first / B follows).

## Still open

- **Survivorship-free rerun** of the monthly picks (`+18.6-monthly-top10-picks.md`) on Sharadar data. This is the only lead left.
- **Forward tests** of the gravity test (`-16.7-brandon-gravity.md`) and of the five published effects run on 2026-10-25 (`-2.2`, `-4.7`, `+0.3`, `+3.7`, `-23.4`) wait on a commit and git tag of their frozen scripts, then on closes after 2026-10-19.
- **Collecting the opening-print effect** in `-16.2-twin-gap-fade.md` would take on-open orders placed from the exchanges' pre-open imbalance data, which is paid data. It is market-making economics rather than a signal, and its large-cap margin was already thin by 2023–2025.
- **The accounting-based strategies, untested.** The best post-publication evidence online is for signals built from accounting and analyst data: profitability, earnings surprises and drift, analyst revisions. None can be tested without point-in-time fundamentals. Sharadar's $39 product covers prices including delisted companies, not financial statements; its fundamentals are a separate, pricier product. Letting SEC requests carry a contact email would give the XBRL financial data free. Both are the user's decision.
- **Blocked without a contact email to SEC:**
  - the 13F ownership check in phantom portfolios;
  - 10-K text after 2020 for imposters, which would allow its unseen-months test;
  - XBRL financial statements, for the accounting-based strategies above.

## Considered and dropped

- **Fund or stocks:** fade a group's moves when its ETF's volume leads them. Partly done, and the evidence points both ways (Brown, Davies & Ringgenberg 2021; Ben-David, Franzoni & Moussawi 2018; against: Kang 2023; Bhojraj, Mohanram & Zhang 2020).
- **Clone-strain gauge:** price gaps between near-identical funds and share classes as a measure of scarce arbitrage money. Close to Pasquariello 2014, and noisy to measure from dividend-adjusted closes.
- **Others:**
  - ticker confusion (Rashes 2001);
  - index-rebalance front-running (a large literature);
  - more outside data (1,753 series already tested twice);
  - new ways to extract daily lead-lag (the census found nothing to extract).
- **From the third round's shortlist:**
  - comomentum timing of momentum (Lou & Polk 2022): it predicts over two years, which gives about nine independent observations here;
  - return seasonality (Heston & Sadka 2008): published and robust, but not about correlation;
  - stocks that hold up when their group sells off: too close to the decoupling test;
  - peer volatility spillover: risk, which the co-movement tools already cover;
  - Wikipedia page views: Wikimedia asks scripts to send contact details, which this project doesn't send.
- **From the fourth round's shortlist of published strategies:**
  - short-term momentum (Medhat & Schmeling 2022): it sorts on share turnover, which needs shares outstanding, and Stooq doesn't have them;
  - volatility-managed market exposure (Moreira & Muir 2017): already the best rule inside `-3.2-market-timing.md`, and real-time versions disappoint (Cederburg et al. 2020);
  - the trend factor (Han, Zhou & Zhu 2016): close to the signal ensemble and formula mining, which found only reversal;
  - the Halloween effect: one observation a year;
  - every strategy built on accounting or analyst data: see "Still open".
