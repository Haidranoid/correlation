# What others have tried

**TL;DR:** Predicting stocks from other stocks' prices has been studied since 1990. Daily lead-lag between liquid US stocks faded through the 1990s and 2000s, and moved into minutes and seconds. What survives in print is slow monthly spillover along economic links, reversal of moves that carried no news, and co-movement as a measure of risk. Published anomalies lose about 58% of their returns after publication. This project's results match that picture.

Compiled 2026-10-21 from web searches (SSRN, NBER, arXiv, journal pages), four novelty checks, and stockai's research report (`stockai/reports/AI models for stock price prediction.md`). Status words: **holds** (still works in later tests), **decayed** (weaker or gone in later data), **costs** (gone after realistic trading costs), **contested** (later work disputes it), **unknown** (no independent later test found).

## 1. Lead-lag between stocks

| Work | What they did | What happened |
|---|---|---|
| Lo & MacKinlay 1990, *RFS* ([link](https://ideas.repec.org/a/oup/rfinst/v3y1990i2p175-205.html)) | Large stocks' weekly returns lead small stocks'; much of contrarian profit is this lead-lag | **Decayed:** daily contrarian profits fell steadily 1995–2007 (Khandani & Lo 2011). The big-to-small lead is still detectable to 2015 but unstable (Kinnunen 2016) |
| Brennan, Jegadeesh & Swaminathan 1993; Badrinath, Kale & Noe 1995; Chordia & Swaminathan 2000 | More analyst coverage, more institutions, more volume → leads | **Unknown** after publication; stronger when large stocks are illiquid (Chordia, Sarkar & Subrahmanyam 2011) |
| Hou 2007, *RFS* ([link](https://ideas.repec.org/a/oup/rfinst/v20y2007i4p1113-1138.html)) | Big firms lead small firms within industries | Reproduces in the original period; post-publication unknown |
| Hou & Moskowitz 2005, *RFS* ([link](https://ideas.repec.org/a/oup/rfinst/v18y2005i3p981-1020.html)) | "Price delay": slow responders to the market earn a premium | **Contested:** the tradable part sits in tiny stocks. This project's cluster result (`-4.5-cluster-lead-lag.md`) is this effect rediscovered |
| Scherbina & Schlusche 2020, *Rev. Finance* ([link](https://ideas.repec.org/a/oup/revfin/v24y2020i1p189-225..html)) | Leader-follower pairs picked by Granger tests on single-stock returns | Out of sample in their data. **The closest published analogue to the census** (`+2.0-correlation-census.md`); no cost test or replication |
| Zeng & Mills 2023, *IJFE* ([link](https://doi.org/10.1002/ijfe.2480)) | Decomposed Granger-identified lead-lag pairs | 95% vanish after standard factors; customer-supplier links explain about 1.2% |
| Chinco, Clark-Joseph & Ye 2019, *JF* ([link](https://www.nber.org/papers/w23933)) | LASSO on all stocks' lagged 3-minute returns | Works at the minute scale (2005–12); predictors live about 15 minutes |
| Huth & Abergel 2014 ([link](https://www.arxiv.org/abs/1111.7103)); Curme et al. 2015 ([link](https://arxiv.org/abs/1401.0462)) | High-frequency lead-lag networks | **Costs:** can't cover the spread. By 2011–12, validated links appear only intraday |
| Rapach, Strauss, Tu & Zhou 2019 | LASSO on all industries' lagged monthly returns | **Contested:** an independent replication gets about 3.5%/yr instead of 8%+ |
| Bennett, Cucuringu & Reinert 2022 ([link](https://arxiv.org/abs/2201.08283)) | Clusters of leaders and followers, daily | Gross only; fades after 2012. Replicated here (`-4.5-cluster-lead-lag.md`): real before 2013, gone since, nothing after costs |
| Cartea, Cucuringu & Jin; Zhang et al. 2023 ([link](https://arxiv.org/abs/2305.06704)) | Lead-lag portfolios, robust lead-lag in factor models | Gross only; one replication reports formula errors |
| Pu, Roberts, Dong & Zohren 2023 ([link](https://arxiv.org/abs/2308.11294)) | Learned network momentum across 64 futures | Sharpe 1.5 (2000–22); futures, not stocks; costs not stated |
| Billio et al. 2012; Diebold & Yilmaz 2014; transfer entropy | Directed networks of influence | Measure systemic risk and connectedness, not the direction of returns |

## 2. Spillovers along economic links (monthly)

| Work | What they did | What happened |
|---|---|---|
| Moskowitz & Grinblatt 1999, *JF* | Industry momentum | **Contested:** subsumed by shared-analyst momentum |
| Cohen & Frazzini 2008, *JF* ([link](https://ideas.repec.org/a/bla/jfinan/v63y2008i4p1977-2011.html)) | Customers' returns predict suppliers' | **Decayed:** value-weighted 1.30% → 0.62%/month after 2005, not significant (Pinchuk 2023) |
| Menzly & Ozbas 2010; Cohen & Lou 2012 | Input-output links; conglomerates follow stand-alone firms | Weaker in replications (Hou, Xue & Zhang) |
| Lee, Sun, Wang & Zhang 2019; Parsons, Sabbatucci & Titman 2020 | Technology links; same-city firms | Both subsumed by shared-analyst momentum |
| Ali & Hirshleifer 2020, *JFE* ([link](https://papers.ssrn.com/abstract=3015582)) | Firms covered by the same analysts | The strongest: 1.19%/month alpha; break-even cost 42.5 bps; weaker later in their sample; no independent replication |
| Hoberg & Phillips 2018, *JFQA* ([link](https://doi.org/10.1017/S0022109018000479)) | Peers from 10-K text, especially those outside the same industry code | Large momentum from less visible peers; no replication found |
| Thomas & Zhang 2008, *JAR* | Peers overreact to early announcers' earnings, then reverse | **Decayed:** gone in recent high-volume years (Chung, Hrazdil & Trottier 2014) |
| Huang, Lee, Song & Xiang 2022, *JFE* ([link](https://www.gsb.stanford.edu/faculty-research/publications/frog-every-pan-information-discreteness-lead-lag-returns-puzzle)) | Leader news arriving gradually vs in jumps | Underreaction only to gradual news |
| Burt & Hrdlicka 2021, *JFQA* | Where linked-firm predictability comes from | Beyond a month it is shared momentum, not slow news: a critique of the whole family |
| Wang 2025/26, *JFQA* ([link](https://doi.org/10.1017/S002210902510238X)) | Peer spillover split into overnight and intraday | Peers' overnight moves carry overnight, then fully reverse intraday, so close-to-close tests miss them. Monthly horizon, analyst-linked peers |
| Feng et al. 2025, *JFE* | Links from bond credit-rating co-movement | About 0.45%/month; new |
| This project, peer momentum (`-5.2-peer-momentum.md`) | Peers from same-day co-movement | Slightly reversed (z = −2.79). Chen et al. 2019 explain why: correlation peers mostly share price pressure that reverts |

## 3. Pairs trading and statistical arbitrage

| Work | What they did | What happened |
|---|---|---|
| Gatev, Goetzmann & Rouwenhorst 2006, *RFS* | Trade the gap between closely matched pairs | **Decayed:** about 11%/yr to 2002, much less after |
| Do & Faff 2010, 2012 | The same rule with full costs | **Costs:** largely unprofitable after 2002 |
| Avellaneda & Lee 2010, *QF* | Mean reversion of residuals against factors or sector ETFs | **Decayed:** net Sharpe 1.44 over 1997–2007 for the principal-components version, falling to about 0.9. The fund version was 1.1, and by year from 2002 to 2007 it fell to −0.2 (replicated here in `-6.8-etf-stat-arb.md`) |
| Chen, Chen, Chen & Li 2019, *Management Science* ([link](https://doi.org/10.1287/mnsc.2017.2825)) | Each stock against its 50 most-correlated stocks | The signal is own-stock reversal plus 1-month industry momentum; after 2008, significant only equal-weighted |
| Jin, Cucuringu & Cartea 2023 ([link](https://doi.org/10.1145/3604237.3626894)) | Cluster residual correlations, trade mean reversion within clusters | Gross Sharpe > 1; modest costs cut it sharply |
| Guijarro-Ordonez, Pelger & Zanotti 2022 ([link](https://arxiv.org/abs/2106.04028)) | Deep learning on residuals of about 550 large caps | Gross Sharpe about 4; net about 1 at 5 bps; no impact costs; no replication |

## 4. Co-movement, crowding and ownership

| Work | What they did | What happened |
|---|---|---|
| Antón & Polk 2014, *JF* ([link](https://doi.org/10.1111/jofi.12149)) | Shared fund ownership → excess co-movement → cross-stock reversal | About 9%/yr, 1980–2008; no later test |
| Lou & Polk 2022, *RFS* ([link](https://doi.org/10.1093/rfs/hhab117)) | Crowding of momentum measured as excess correlation | High crowding → momentum reverses; **holds** in their tests |
| Coval & Stafford 2007; Wardlaw 2020, *JF* | Fire sales by funds with outflows | **Contested:** the standard measure contains the stock's own return; corrected, no reversal |
| Sias, Turtle & Zykaj 2016 | Hedge-fund crowd demand | Predicts *higher* returns, no reversal |
| Kinlaw, Kritzman & Turkington 2019; Baltas 2019 | Crowding from correlation structure | Crowded sectors and momentum-type premia underperform |
| Pollet & Wilson 2010; Kritzman et al. 2011 | Market-wide average correlation; the absorption ratio | Predict the market premium and drawdown risk, not single stocks |
| Barberis, Shleifer & Wurgler 2005; Da & Shive 2018 | Index inclusion and ETFs create co-movement | Real; partial reversal; no tradable net test |
| Greenwood & Thesmar 2011 | Ownership fragility | Predicts volatility, not direction (**holds**) |
| Asness, Frazzini, Gormsen & Pedersen 2020 | Betting against correlation | Low-correlation stocks earn more (**holds** in the authors' tests) |

## 5. Machine learning and AI

| Work | What they did | What happened |
|---|---|---|
| Gu, Kelly & Xiu 2020, *RFS* | Trees and neural nets on about 900 inputs, monthly | Double linear models' performance; **contested:** gains concentrate in microcaps and distressed stocks, and shrink after costs (Avramov, Cheng & Metzker 2023) |
| Jiang, Kelly & Xiu 2023, *JF* | CNNs on price-chart images | Gross Sharpe 7.2 equal-weighted, 1.7 value-weighted; gross only |
| StockGPT 2024; Kronos 2025; FinText-TSFM 2025 (from stockai's report) | GPT-style models on return tokens | 51–52% daily direction; gross long-short Sharpe 4–8. **Every daily strategy in the largest study turns negative at 20 bps.** CatBoost matches them |
| Zero-shot foundation models (Chronos, TimesFM, Toto) | Forecast returns with no finance training | Worse than predicting zero (negative R²) |
| Tan et al. 2024 (NeurIPS) | Deleted the language model from LLM forecasters | No loss of accuracy in most comparisons |
| Lopez-Lira & Tang 2023 ([link](https://arxiv.org/abs/2304.07619)) | GPT scores of headlines | Gross Sharpe about 3, decaying as adoption rises; unprofitable at 20 bps round trip |
| Lopez-Lira, Tang & Zhu 2025 | LLM memorization | GPT-4o "forecasts" the S&P 500 with 80% accuracy before its cutoff and 46% after: LLM backtests on known history are meaningless |
| Graph neural network surveys (Patel et al. 2024) | Industry, supplier and correlation graphs | Accuracy gains reported; net Sharpe almost never |
| TDA (Gidea & Katz 2018); network curvature (Sandhu et al. 2016); Hawkes jumps (Aït-Sahalia et al. 2015) | Shape of the correlation network | Crash-risk and jump-clustering indicators, not direction |
| This project (stockai's stockGPT and LightGBM files) | stockGPT and LightGBM, graded against random accounts | Real gross signals, mostly reversal and a low-volatility lean; costs eat them; the monthly lead (`+18.6-monthly-top10-picks.md`) awaits survivorship-free data |

## 6. Overnight, intraday and reversal

| Work | What they did | What happened |
|---|---|---|
| Lou, Polk & Skouras 2019, *JFE* ([link](https://eprints.lse.ac.uk/87481/)) | 14 anomalies split into overnight and intraday | Momentum (including industry momentum) earns overnight; most others intraday |
| Barardehi, Bogousslavsky & Muravyev (RFS, forthcoming) | A stock's own past day and night returns | Intraday returns drive short-term reversal and momentum |
| Berkman, Koch, Tuttle & Zhang 2012, *JFQA* | Attention-grabbing stocks | Inflated opens reverse intraday: buying at the open is costly |
| Nagel 2012, *RFS* ([link](https://www.nber.org/papers/w17653)) | Reversal profits vs VIX | Reversal pays for providing liquidity; larger when liquidity is scarce |
| Da, Liu & Schaumburg 2014; Blitz et al. 2013 | Reversal of residuals (news and industry removed) | 2–4× plain reversal; net of costs in large caps (Blitz et al.) |
| Medhat & Schmeling 2022, *RFS* | Last month's return × turnover | Low turnover reverses, high turnover continues (**holds**) |
| Chan 2003; Savor 2012; Boudoukh et al. 2019 | Own-stock moves with and without news | No-news moves reverse; news moves drift |
| Bogousslavsky & Muravyev 2023, *JFM* | Who trades at the close | Closing auctions have lower impact, but prices usually print at the pre-close bid or ask, so auction orders aren't free |

## 7. Outside data

| Work | What they did | What happened |
|---|---|---|
| Welch & Goyal 2008; Goyal, Welch & Zafirov 2024, *RFS* | 17, then 46, macro and valuation predictors of the market | None beats the historical mean out of sample; newer ones mostly fail too (**decayed**) |
| Preis, Moat & Stanley 2013; Challet & Bel Hadj Ayed 2013 | Google Trends trading rules | **Contested:** random keywords do as well |
| Bollen, Mao & Zeng 2011; Lachanski & Pav 2017 | Twitter mood predicts the Dow | **Failed replication** |
| Driesprong, Jacobsen & Maat 2008 | Oil predicts the market a month later | Not re-verified here; this project's outside-data test (`+4.0-outside-data-leaders.md`) found nothing for oil or 1,752 other series |

## 8. Decay, costs and replication

| Work | Finding |
|---|---|
| McLean & Pontiff 2016, *JF* | Anomaly returns are 26% lower out of sample and 58% lower after publication |
| Chordia, Subrahmanyam & Tong 2014 | Anomaly returns roughly halved after decimalization (2001) |
| Harvey, Liu & Zhu 2016, *RFS* | With hundreds of factors tried, a new one needs t > 3 |
| Hou, Xue & Zhang 2020, *RFS* | 65% of 452 anomalies fail value-weighted; 96% of trading-friction anomalies fail |
| Jensen, Kelly & Pedersen 2023, *JF* | Most factors replicate statistically, which isn't the same as profitable after costs |
| Novy-Marx & Velikov 2016, *RFS* | Low-turnover anomalies survive costs; reversal-type ones mostly don't |
| Chen & Velikov 2020 | After costs and publication, the average anomaly earns about 8 bps a month |
| Frazzini, Israel & Moskowitz (AQR) | Real institutional costs are well below academic estimates; short-term reversal is the most cost-constrained |

## 9. Nearest prior work for the last ten ideas (tested 2026-10-21)

From the separate novelty checks; each experiment file has the full note.

| Work | What they did | What happened, there and here |
|---|---|---|
| Yang & Yin 2024; Wang 2025/26, *JFQA* | Day-versus-night beta; peers' overnight moves | Monthly. Daily here (`-6.0-night-and-day.md`): the catch-up to the market is real, but since 2020 it comes overnight, out of a daily trader's reach |
| Della Corte, Kosowski & Wang 2015 (SSRN 2730304) | Each day, buy the lowest close-to-open returns and short the highest, from open to close | 1.68% a day gross in the US, 1993–2014, mostly in the first minute. Against the twin here (`-16.2-twin-gap-fade.md`): +74 bps a day from the official open, a fifth of it left at 9:35, the same finding |
| Bogousslavsky & Muravyev 2023; Johnston (Purdue dissertation) | Closing prices sit away from the midpoint; a high-low anomaly | Here (`+2.5-close-location.md`): nothing beyond 5-day reversal |
| Cohen, Malloy & Nguyen 2020, *JF* | Companies that rewrite their 10-K underperform | 34–58 bps a month, 1995–2014. Here (`-1.3-lazy-prices.md`): t 0.26 over 2007–2021 |
| Pollet & Wilson 2010 | Average correlation predicts the market premium | Here (`-3.2-market-timing.md`): no better than random timing; volatility-managed SPY did better |
| Blitz, Hanauer, Honarvar, Huisman & van Vliet 2023, *FAJ* | Five short-term signals combined, with a buy-and-hold buffer | Over 6% a year after costs on MSCI World. Here (`-3.2-signal-ensemble.md`): IC t 3.0, but the long-short nets t 0.27 |
| Stowe & Xing 2010; Pozzi, Di Matteo & Aste 2013 | Low-R² stocks; stocks on the edge of correlation networks | Here (`-4.3-loners.md`): t 1.5, and lost 23% in the unseen months |
| Blitz, Huij & Martens 2011; Hanauer & Windmüller 2023 | Residual momentum; its decay in the US | Against the twin here (`-0.1-twin-momentum.md`): the hedge cuts risk, but it earns no more than plain momentum, itself zero among the 500 largest |
| Yan & Zheng 2017; Chordia, Goyal & Saretto 2020 | Mining thousands of signals, with controls for luck | Here (`-2.4-formula-mining.md`): of 1,352 price and volume formulas, the best were all reversal, faded in the holdout |
| Zaremba, Szyszka, Karathanasopoulos & Mikutowski 2021 | Breadth predicts market and industry returns in 64 countries | No test of a thrust trigger. Here (`+2.6-breadth-thrust.md`): +15.5 bps a month after thrusts, t 1.3 |

## 10. Nearest prior work for the third ten ideas (tested 2026-10-21)

From the separate novelty checks; each experiment file has the full note.

| Work | What they did | What happened, there and here |
|---|---|---|
| Lachance 2023, *RFE*; Aboody et al. 2018; Lou, Polk & Skouras 2019 | Hold persistent-overnight stocks only overnight, through the auctions | Lachance reports large gross returns to 2014; the NightShares overnight ETFs closed in 2023. Against the twin here (`+15.7-overnight-auctions.md`): +11.6 bps a day gross, about +1 after 2 bps a side, losing since 2020. The profit sits in the opening print |
| Wang 2025/26, *JFQA*; Lu, Zhang, Reinert & Cucuringu 2025 | Overnight-versus-daytime spillovers and lead-lag networks | Peers chosen by overnight correlation here (`-1.6-overnight-peers.md`) are real economic peers (Apple's are its suppliers), but their past month doesn't predict: −0.31% a month, t −0.9 |
| Hoberg & Phillips 2018, *JFQA*; Ali & Hirshleifer 2020 | Momentum from less visible text peers; shared-analyst momentum subsumes it | Here (`+4.6-hidden-text-peers.md`): text peers that don't co-move don't predict (t −0.33), and text-industry momentum as a whole is absent |
| Ehsani & Linnainmaa 2022, *JF*; Falck, Rej & Thesmar 2020 | Factor momentum in named factors; it adds little beyond stock momentum without the last month | On eigenportfolios of stock returns here (`-3.8-factor-momentum.md`): +0.07% a month net, 0.73 correlated with plain momentum |
| Kirby & Ostdiek 2012, *JFQA*; Jain & Jain 2019; Scherer 2011 | Does covariance information beat volatility-only weights? Mostly no, after costs | Here (`-4.6-correlation-portfolios.md`): minimum variance has a quarter to a third less risk than inverse volatility in every period, but no better Sharpe |
| Chinco, Clark-Joseph & Ye 2019, *JF*; Cartea, Cucuringu & Shi 2026 | Minute-scale cross-stock predictability | On 2026 Yahoo bars here (`-100.0-intraday-lead-lag.md`): 5-minute lead-lag at chance. The 1-minute lead-lag is nonsynchronous trading and breaks even at 0.45 bp a side |
| Wang, Yan & Zheng 2020, *JFE*; Ardia, Aymard & Cenesizoglu 2024 | FINRA daily short volume, 2010–2015: the level predicts and the abnormal ratio doesn't. Retail imbalance stopped predicting large caps after 2015 | Against the twin group here (`+0.0-short-volume.md`): a rising ratio predicts slightly lower returns (sign learned on 2010–2015; IC t 2.9 in 2016–2019, t 1.1 once lagged a day), but the weekly book nets −0.5 bps a week after costs |
| Wahal & Yavuz 2013, *JFE*; Chen, Singal & Whitelaw 2016 | Style comovement; most "new comovement" is momentum | Here (`-3.5-theme-joiners.md`): joining a hot data-driven group predicts slightly *lower* returns after own-return controls (t −0.65) |
| Engelberg, Gao & Jagannathan 2009; Da, Gurun & Warachka 2014 | Divergences on firm news persist; gradual news drifts | Here (`-10.2-decoupling.md`): after a sustained break from the twin, the drift slightly reverses (−82 bps, t −0.8). The events are mostly market-wide correlation shifts |
| Savor & Wilson 2014, *JFE*; Cui, Zhao & Cui 2023; Kurov, Wolfe & Gilbert 2021 | Beta is paid on announcement days; the strategy is published; the pre-FOMC drift faded after 2015 | Here (`-0.6-announcement-beta.md`): the beta slope is +23 bps higher on announcement days in 2006–2019 (t 3.1, FOMC strongest), but the strategy nets +9.5 bps a day, t 1.0. It is gone since 2020 and reversed in the unseen months |

## 11. Ten published strategies replicated here (round four, 2026-10-21)

Each one is tested on 2006–2026, after or mostly after the paper's sample. The paper's figures are as the experiment files cite them; figures marked unverified there are unverified here.

| Paper | Their sample and result | Here, 2006–2026 |
|---|---|---|
| Moskowitz, Ooi & Pedersen 2012, *JFE*; Hurst, Ooi & Pedersen 2017 | Time-series momentum: Sharpe above 1 in futures, 1985–2009; 0.76 since 1880, but 0.41 in 2010–2016 | `+1.2-trend-following.md`: Sharpe 0.43 in 2008–2019 (t 1.5) on 23 funds, matching AQR's live series (0.40 since 2010). It gained in 2008 and 2022 |
| Heston & Sadka 2008; Keloharju, Linnainmaa & Nyberg 2016 | Same-month seasonality: 1.19% a month (t 6.3) over 1963–2011, but 0.40% (t 1.35) in 2003–2011 | `-0.2-seasonality.md`: −0.23% a month (t −1.05); neither half of the pattern |
| Frazzini & Pedersen 2014, *JFE*; Novy-Marx & Velikov 2022 | BAB: 0.70% a month (t 7.1) over 1926–2012; the critique says the construction carries it | `-4.4-betting-against-beta.md`: tracks AQR's series (r 0.67), but +0.25% a month (t 0.9) on liquid stocks, losing since 2020. AQR's own live series: +0.50% (t 2.6) since 2013 |
| Barroso & Santa-Clara 2015, *JFE* | Scaling momentum by its volatility nearly doubles its Sharpe, 1927–2011 | `+3.3-risk-managed-momentum.md`: the rule reproduces on French's UMD (0.49 → 0.94). Here it lifts Sharpe −0.26 → +0.21 and cuts the 2009 crash from −78% to −22%, but the book earns t 0.66 |
| George & Hwang 2004, *JF* | The 52-week high: 0.45% a month (t 2.0), 1963–2001 | `-7.4-52-week-high.md`: −0.38% a period (t −0.97), and it's momentum (R² 0.75). Chen & Zimmermann find 0.18% (t 0.56) after publication |
| Gatev, Goetzmann & Rouwenhorst 2006, *RFS* | Pairs trading: up to about 11% a year, 1962–2002, declining | `+0.8-pairs-trading.md`: zero in 2006–2019 (t 0.04), below random pairs; a small revival in turbulent stretches since 2020 |
| Avellaneda & Lee 2010, *QF* | Fund-residual stat arb: Sharpe 1.1, 1997–2007, falling to −0.2 by 2007 | `-6.8-etf-stat-arb.md`: Sharpe 1.24 before costs, 0.24 after (t 0.8), negative when filled at the next close |
| Bali, Cakici & Whitelaw 2011, *JFE* | Low-minus-high MAX: 1.03% a month (t 2.8, value-weighted), 1962–2005 | `-3.6-max-lottery.md`: wrong sign (t −0.9). Chen & Zimmermann's survivor-free version fell to about 0.2% since publication |
| McConnell & Xu 2008, *FAJ*; Lakonishok & Smidt 1988 | The turn of the month: 0.15% a day (t 8.1), all of the market's excess return, 1926–2005 | `-9.6-turn-of-month.md`: the same code reproduces 1926–2005 (0.147%, t 8.2), then 2006–2025 is flat (t −0.07) |
| Da, Gurun & Warachka 2014, *RFS*; Goyal, Jegadeesh & Subrahmanyam 2025 | Momentum is stronger when information arrives gradually, 1927–2007; robust out of sample internationally | `+2.0-frog-in-the-pan.md`: continuous-information momentum −0.71% a month (t −1.05), and worse than discrete; the six-month spread −2.7% against the paper's +8.0% (t 8.5). Goyal et al.'s own US table shows nothing for 1993–2020 (unverified) |

## Patterns

- **What survives:** slow, monthly spillovers along economic links measured from non-price data, and selective reversal (of residuals, of low-turnover moves, of no-news moves). Co-movement measures mostly predict *risk* (volatility, crashes), not direction.
- **This project's results are the documented ones:**
  - the reversed peer signal (`-5.2-peer-momentum.md`) is what correlation-chosen peers do;
  - "the market's slow followers" (`-4.5-cluster-lead-lag.md`) is price delay;
  - the outside-data null (`+4.0-outside-data-leaders.md`) is what replication studies found;
  - the "gone since 2012" timeline matches the literature's shift of lead-lag into intraday trading.
- **Survivor-only data cuts both ways.** It drops the small delisted stocks where lead-lag concentrates, which hides signal. It also drops delisting losses, which flatters strategies that buy losers: reversal, and the monthly picks (`+18.6-monthly-top10-picks.md`).
- **Gaps** (from the four novelty checks):
  1. **Conditioning on information arrival.** Nobody conditions daily lead-lag on the leader's own news or abnormal-volume days, or on whether the follower had news of its own. → tested in `-7.5-bystanders.md`: no give-back; co-movers follow the news stock's later path
  2. **Inferring crowd holdings from returns** and trading daily unwind events; 13F has been used, returns alone haven't. → tested in `-9.1-phantom-portfolios.md`: unwinds keep falling instead of bouncing
  3. **The gap between text-based and co-movement-based peers** as a continuous trading signal. → tested in `+0.1-imposters.md`: the gap closes over a year, but prices don't correct
  4. **Daily overnight/intraday split of cross-stock lead-lag, executed through the auctions.** Wang 2025/26 did it monthly → tested in `-6.0-night-and-day.md`: the catch-up is real, but since 2020 it happens overnight
  5. **Honest post-2012, net-of-cost evaluations** of network and machine-learning lead-lag are themselves rare. Most papers report gross results ending 2019–22.
