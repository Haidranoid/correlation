# The triple pendulum: why correlations don't predict the market

<p align="center">
  <img src="pendulum.svg" width="500" alt="Three triple pendulums start a billionth of a radian apart, move as one for about ten seconds, then follow completely different paths inside the same circle">
</p>
<p align="center"><sub>Three identical triple pendulums, started a billionth of a radian apart. The animation is <a href="pendulum.svg">pendulum.svg</a>; for the interactive version, download <a href="pendulum.html">pendulum.html</a> with <a href="pendulum.js">pendulum.js</a> and open it in a browser (GitHub doesn't run JavaScript).</sub></p>

**TL;DR:** A triple pendulum follows Newton's laws exactly, yet copies started a billionth of a radian apart end up in completely different places within ten seconds. Stocks share that with it, for different reasons (see "Where the metaphor breaks"). Correlation measures the rods: how stocks move together right now, which is real and useful for risk. A trade needs the swing, where things go next, and that can't be predicted beyond a short horizon that faster traders already own.

## The intuition

Stocks visibly move together. Banks fall together, chipmakers rally together, an oil shock moves every energy stock. Twenty years of daily prices for thousands of stocks hold millions of these relationships. It seems that some of them must say something about tomorrow: if A and B always move together, and A moved today, surely B is more likely to follow.

The project tested that idea, and the ideas that grew from it, against random accounts that hold as many stocks and trade as often. The tests covered:
- every pair of stocks at every lag;
- groups that lead other groups;
- peers chosen by news, text and overnight moves;
- machine learning;
- 1,753 outside data series;
- ten of the best-documented published strategies, on the years after their papers.

Nothing makes money reliably after costs (the full list is in [`../experiments/README.md`](../experiments/README.md)). The pendulum explains why none of this is a surprise.

## The pendulum

Hang a rod from a hinge, hang a second rod from the end of the first, and a third from the end of the second. Let go from high up. The motion follows exact, simple laws, and the simulation here ([`pendulum.js`](pendulum.js)) solves them to within a few parts in a billion. Even so:

| Measured in this simulation | |
|---|---|
| Copies started 0.000000001 radians apart are visibly different after | 9.6 seconds |
| The gap between them doubles every | 0.33 seconds |
| Measuring the start a billion times more precisely (0.001 → 0.000000000001 rad) extends the forecast from | 3.5 to 11.6 seconds |
| Change in total energy over 30 seconds | 3 parts in a billion |

This is chaos in the mathematical sense. The future is fully determined, but any error in the present grows exponentially. A forecast's horizon grows only with the *logarithm* of your precision, so better data buys almost nothing.

Here that chaos is measured, not proven: the numbers above come from the simulation. Proving that even a simple mechanical system is chaotic across a whole region of starting points is hard. For the standard map, a pendulum kicked at regular intervals, it was a long-standing open problem (Sinai's conjecture) until a 2026 proof in OpenAI's machine-generated [`openai/math`](https://github.com/openai/math) collection (family 146, checked in the Lean proof assistant, not yet peer-reviewed).

Read the three rods as the three layers of a stock's move:

- **The first rod is the market:** everything moves with it.
- **The second rod is the sector:** the fund that tracks the stock best. This project calls it the stock's twin.
- **The third rod is the company itself.**

The tip, where the stock price ends up, is the sum of all three swings.

## What you can predict, and what you can't

| In the pendulum | In the market | What the experiments found |
|---|---|---|
| **The rods.** The bobs are always joined; their relationship at any instant is exact. | Same-day co-movement: which stocks move together, and how much. | Real and stable. It persists year to year and builds useful tools ([`+5.1-co-movement-tools`](../experiments/+5.1-co-movement-tools.md)). |
| **The energy.** The size of the swings is conserved. | Volatility: how much stocks move, and together. | Predictable. Minimum-variance portfolios built from correlations had 63–76% of the risk of portfolios that ignore them, in every period tested ([`-4.6-correlation-portfolios`](../experiments/-4.6-correlation-portfolios.md)). |
| **The reach.** The tip never leaves the circle. | Drawdowns and how far a portfolio can fall. | In 2008 the correlation-built portfolio fell 36%, the volatility-only one 52%. |
| **The tip's position in ten seconds.** | Tomorrow's direction. | Chance. Lagged correlations between stocks are at chance level at every lag and move length tried ([`+2.0-correlation-census`](../experiments/+2.0-correlation-census.md), [`-1.9-unequal-move-lengths`](../experiments/-1.9-unequal-move-lengths.md)). |
| **The hinge passing motion along.** One bob moves, and the next responds almost at once. | Lead-lag: one stock or fund moving first. | It exists, but within a minute. On 2026's 1-minute bars, funds lead stocks; at 5 minutes it is gone, and trading it loses money ([`-100.0-intraday-lead-lag`](../experiments/-100.0-intraday-lead-lag.md)). |

## Five reasons the edge isn't there

**1. Correlation describes the rods, not the swing.**
- A correlation is measured between moves that happen *at the same time*. It says that when the sector swings, the stock swings with it. It doesn't say which way the sector will swing tomorrow.
- In the pendulum, knowing that bob two is attached to bob one tells you nothing about where bob one is going.
- The project's census of every pair, offset and move length found lagged relationships at exactly the rate chance produces. Same-day relationships were strong and repeated year after year.

**2. The forecast horizon is shorter than a trading day.**
- In the pendulum, the motion is predictable for a few seconds. In the market, the window is minutes.
- Every effect the project found that was large enough to look like money lived at the open, inside the first few minutes:
  - fading each stock's opening gap made 74 bps a day from the official opening price, and kept only a fifth of it at 9:35 ([`-16.2-twin-gap-fade`](../experiments/-16.2-twin-gap-fade.md));
  - overnight drift showed the same pattern ([`+15.7-overnight-auctions`](../experiments/+15.7-overnight-auctions.md)).
- A strategy that decides on daily data looks at the pendulum once a day. By then the predictable part of the swing has already happened.

**3. Someone else is watching with a faster camera, and pushing.**
- A real pendulum doesn't care whether you predict it. A market is moved by people who are all trying to.
- Anything predictable at a horizon you can trade gets traded until it stops being predictable. Published anomalies lose about 58% of their returns after publication (McLean & Pontiff 2016).
- This project watched it happen:
  - the beta premium on Fed, inflation and jobs days replicated over 2006–2019 (t 3.1) and was gone after 2020 ([`-0.6-announcement-beta`](../experiments/-0.6-announcement-beta.md));
  - so was the small signal in FINRA's short-volume data ([`+0.0-short-volume`](../experiments/+0.0-short-volume.md));
  - the group lead-lag of Bennett et al. replicated before 2013 and vanished after ([`-4.5-cluster-lead-lag`](../experiments/-4.5-cluster-lead-lag.md)).
- The clearest case is the turn of the month. On 1926–2005 data, the market earned all of its excess return in the four days around each month's turn, and this project's code reproduces that exactly (t 8.2). On 2006–2025 the same code finds nothing (t −0.07) ([`-9.6-turn-of-month`](../experiments/-9.6-turn-of-month.md)).
- Trend following, the strategy with the longest record of all, still behaves the same way, but its Sharpe ratio fell from 1.4 over the original paper's sample to 0.4 since 2010 ([`+1.2-trend-following`](../experiments/+1.2-trend-following.md)).

**4. Touching the pendulum costs more than the edge.**
- To use a forecast you have to trade, and trading costs a few basis points each way. Several signals beat random accounts before costs:
  - short-term reversal (+11.8 points a year);
  - a LightGBM model on next-day returns (+35.8).
- Each trades so often that the costs take more than the edge. The intraday lead-lag breaks even only below half a basis point per trade.
- In the pendulum, every measurement nudges it. In the market every trade does, and the nudge is bigger than the signal.

**5. Twenty years is fewer swings than it looks.**
- Twenty years of daily data sounds like 5,000 observations per stock, but the market has a handful of genuinely independent episodes in that time: 2008, 2020, 2022 and a few rotations.
- Many results in this project were mostly one crisis. With 74 tests at the usual bar, three or four false passes are expected by luck, which is why every pass here had to survive 16 months of data it had never seen.
- In pendulum terms, you've watched only a few of its swings, and each one went somewhere new.

## What does work: knowing the reach

The pendulum's energy and reach are as predictable as its position is not. The same is true here:
- **The co-movement tools,** which answer "what moves with what?", gave advice that beat random choices out of sample. They cut a portfolio's next-year volatility by 5.1%, and the top suggested hedge left 68% of a stock's risk against 91% for a random fund.
- **Correlation-built portfolios** were a quarter to a third calmer than portfolios that ignore correlations, in every period including the 16 unseen months.
- **Published strategies show the same split.** Scaling momentum by its own recent volatility cut its 2009 crash from −78% to −22%, but didn't make momentum pay ([`+3.3-risk-managed-momentum`](../experiments/+3.3-risk-managed-momentum.md)). Trend following gained in 2008 and 2022, when stocks and bonds fell, but earned little on average.

That is a real edge, but in risk, not return. You can't say where the tip will be. You can say how far it can reach, and build something that survives the swing.

## Where the metaphor breaks

The pendulum gets the lesson right: knowing a system's structure exactly does not give you its future. It gets the reason partly wrong, in four ways.

- **Markets aren't chaotic in the pendulum's sense.**
  - A pendulum is deterministic. Measure its start perfectly and you could predict it forever; it becomes unpredictable only because tiny measurement errors grow exponentially.
  - Researchers looked for that kind of low-dimensional chaos in stock prices in the 1990s (Hsieh 1991, *Journal of Finance*, is the classic). Mostly they found that the non-randomness in returns is volatility clustering, not chaos.
  - Prices are unpredictable mainly because news arrives at random. More precise measurement wouldn't help.
- **The market fights back.**
  - A pendulum doesn't care whether you predict it. Traders do, and any pattern that can be seen gets traded until it's gone.
  - This project watched it happen. The turn-of-the-month effect held from 1926 to 2005 (t 8.2) and has been flat since. Trend following's Sharpe ratio fell from 1.4 to 0.4 after it was published.
  - This, not chaos, is the main reason no edge survived.
- **It's too generous about the short term.** A pendulum is very predictable for a few seconds. The market isn't: the one-minute lead-lag found here was mostly a timing artifact, the busiest stocks' prices updating first, and nobody can collect it after costs.
- **The rods aren't rigid.**
  - Correlations jump in crises, and only about half of each co-movement group stays together from one year to the next.
  - Same-day co-movement is far more stable than any lagged relationship, but it isn't fixed the way a rod is.

**A more exact description** is a coin flip where the size of the bet is predictable but the side isn't:
- volatility and co-movement can be forecast;
- the direction is close to 50/50;
- whenever someone finds a slightly weighted coin, the crowd flattens it.

Weather and climate make another fair comparison: no one can forecast next month's weather, but you can know the climate, which is the risk.

The pendulum stays because it makes the one part that transfers exactly obvious at a glance: perfect knowledge of how the pieces are connected tells you almost nothing about where they go next.

## Files

| File | What it is |
|---|---|
| [`pendulum.svg`](pendulum.svg) | The animation above: SVG with built-in (SMIL) animation, so it plays on GitHub. Generated, don't edit |
| [`pendulum.html`](pendulum.html) | The interactive version: change the starting difference, switch between a big (chaotic) and small (orderly) swing, and watch how far apart the tips are on a log scale |
| [`pendulum.js`](pendulum.js) | The physics, shared by both: exact equations of motion for point masses on massless rods, fourth-order Runge-Kutta with a 0.5 ms step |
| [`render-svg.mjs`](render-svg.mjs) | `node render-svg.mjs` (Node 18 or later, no packages) rebuilds `pendulum.svg` and prints the numbers in the table above |

**The physics, briefly.** With angles θ measured from straight down and μᵢⱼ the mass hanging below hinge max(i, j), each hinge obeys

Σⱼ μᵢⱼ lᵢ lⱼ [cos(θᵢ − θⱼ) θ̈ⱼ + sin(θᵢ − θⱼ) θ̇ⱼ²] + g μᵢᵢ lᵢ sin θᵢ = 0

The code solves this 3×3 linear system for the accelerations at every step. Energy conservation is the accuracy check. The separation between copies grows as e^(2.1 t) while it is small, so it doubles every 0.33 seconds.
