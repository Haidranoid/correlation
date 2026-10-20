//! `c2 portfolio`, `c2 diversify`, `c2 hedge`, `c2 groups`: the same-day co-movement tools
//! (`c2_census::together`) as text.

use anyhow::Result;
use c2_census::together::{self, Candidate, HedgeRow, Universe, words};

fn universe(as_of: &Option<String>, years: usize) -> Result<Universe> {
    let store = c2_data::Store::open(&super::store_dir())?;
    let as_of = as_of.as_deref().map(c2_data::parse_date).transpose()?;
    Universe::build(&store, as_of, years)
}

fn pct(v: f64) -> String {
    format!("{:.1}%", 100.0 * v)
}

/// "0.55 often together".
fn corr_words(r: f64) -> String {
    format!("{r:+.2} {}", words(r))
}

fn before(r: Option<f64>) -> String {
    r.map(|r| format!("{r:+.2}")).unwrap_or_else(|| "new".into())
}

fn span(d: (i32, i32, usize)) -> String {
    format!("{} .. {} ({} days)", c2_data::date_string(d.0), c2_data::date_string(d.1), d.2)
}

fn span2(d: Option<(i32, i32)>) -> String {
    d.map(|d| format!("{} .. {}", c2_data::date_string(d.0), c2_data::date_string(d.1))).unwrap_or_else(|| "not in the data".into())
}

fn holdings_line(p: &together::PortfolioReport) -> String {
    let n = p.holdings.len();
    format!(
        "Your {n} holding{} behave{} like {:.1} independent bet{}.",
        if n == 1 { "" } else { "s" },
        if n == 1 { "s" } else { "" },
        p.bets,
        if (p.bets - 1.0).abs() < 0.05 { "" } else { "s" }
    )
}

fn print_portfolio_head(p: &together::PortfolioReport) {
    println!("Daily returns {}; a year earlier: {}.", span(p.days), span2(p.before_days));
    if let Some(why) = &p.shortened {
        println!("Note: {why}, so only those days are used.");
    }
    println!("{}", holdings_line(p));
    println!(
        "Volatility {} a year for the portfolio, against {} for its holdings on average (by weight). Beta to SPY {:.2}; correlation with SPY {}.",
        pct(p.vol),
        pct(p.avg_vol),
        p.beta,
        corr_words(p.spy_corr)
    );
}

pub fn portfolio(holdings: &[String], as_of: &Option<String>) -> Result<()> {
    let h = together::parse_holdings(&holdings.join(" "))?;
    let u = universe(as_of, 2)?;
    let p = together::portfolio(&u, &h)?;
    println!("Portfolio check (same-day co-movement).");
    print_portfolio_head(&p);
    println!("\n{:<8} {:<6} {:>7} {:>10} {:>6} {:>13}  {}", "holding", "kind", "weight", "volatility", "beta", "share of risk", "with the rest of the portfolio");
    for x in &p.holdings {
        println!("{:<8} {:<6} {:>7} {:>10} {:>6.2} {:>13}  {}", x.ticker, together::kind(x.etf), pct(x.weight), pct(x.vol), x.beta, pct(x.risk_share), corr_words(x.corr_rest));
    }
    let dups = p.duplicates(0.8);
    println!("\nNear-duplicates (correlation 0.8 or more; a year earlier in brackets):");
    if dups.is_empty() {
        println!("  none");
    }
    for (a, b, r, was) in dups {
        println!("  {} and {}: {} ({})", p.holdings[a].ticker, p.holdings[b].ticker, corr_words(r), before(was));
    }
    let k = p.holdings.len();
    if (2..=12).contains(&k) {
        println!("\nCorrelations over the last year (a year earlier in brackets):");
        print!("{:<8}", "");
        for x in &p.holdings {
            print!(" {:>13}", x.ticker);
        }
        println!();
        for a in 0..k {
            print!("{:<8}", p.holdings[a].ticker);
            for b in 0..k {
                if a == b {
                    print!(" {:>13}", "-");
                } else {
                    print!(" {:>13}", format!("{:+.2} ({})", p.corr[a * k + b], before(p.corr_before[a * k + b])));
                }
            }
            println!();
        }
    } else if k > 12 {
        println!("\n(The full correlation table of {k} holdings is in the GUI's Together tab.)");
    }
    println!("\nScale: 0.9 or more almost always together, 0.6 usually, 0.3 often; -0.3 to 0.3 mostly independent; below -0.3 often opposite, -0.6 usually, -0.9 almost always.");
    Ok(())
}

fn candidate_table(title: &str, rows: &[Candidate], vol_now: f64, add: f64) {
    println!("\n{title}");
    if rows.is_empty() {
        println!("  none");
        return;
    }
    println!("{:<8} {:<6} {:>10}  {:<32} {:>14} {:>17}  {}", "", "kind", "volatility", "with the portfolio", "a year earlier", format!("vol. with {:.0}% in", 100.0 * add), "moves almost exactly like it");
    for c in rows {
        let alike = match c.alike.len() {
            0 => String::new(),
            n if n <= 4 => c.alike.join(", "),
            n => format!("{} and {} more", c.alike[..3].join(", "), n - 3),
        };
        println!(
            "{:<8} {:<6} {:>10}  {:<32} {:>14} {:>9} ({:+.1})  {}",
            c.ticker,
            together::kind(c.etf),
            pct(c.vol),
            corr_words(c.corr),
            before(c.corr_before),
            pct(c.vol_after),
            100.0 * (c.vol_after - vol_now),
            alike
        );
    }
}

pub fn diversify(holdings: &[String], weight: f64, max_vol: f64, show: usize, as_of: &Option<String>) -> Result<()> {
    let h = together::parse_holdings(&holdings.join(" "))?;
    let u = universe(as_of, 2)?;
    let p = together::portfolio(&u, &h)?;
    let d = together::diversifiers(&u, &p, weight / 100.0, max_vol / 100.0, show);
    println!("Best diversifiers: the stocks and funds that move least with your portfolio on the same day, lowest correlation first. Correlation doesn't change with size or leverage, so a 3x fund ranks with its 1x version; funds that move almost exactly alike are shown once, under the least volatile.");
    print_portfolio_head(&p);
    println!("The last columns show the portfolio's volatility with {weight}% of the money moved into each.");
    candidate_table("Funds:", &d.funds, p.vol, d.add);
    candidate_table("Single stocks:", &d.stocks, p.vol, d.add);
    candidate_table("Moving opposite (hedges rather than diversifiers: correlation -0.3 or lower with the portfolio, or -0.5 or lower with SPY, such as inverse funds):", &d.hedges, p.vol, d.add);
    println!(
        "\nLooked at {} liquid stocks and funds (a close on 90% of days, last close $5 or more, median dollar volume $5M or more). Left out: {} cash-like ones (under 3% a year volatility) and {} very volatile ones (over {:.0}% a year, mostly leveraged funds; --max-volatility changes the limit).",
        d.scanned, d.cash_like, d.too_volatile, 100.0 * d.max_vol
    );
    Ok(())
}

fn hedge_table(title: &str, rows: &[HedgeRow]) {
    println!("\n{title}");
    println!("{:<8} {:<6} {:>10}  {:<32} {:>12} {:>12} {:>10}", "", "kind", "volatility", "same day", "a year earlier", "hedge ratio", "risk left");
    for r in rows {
        println!("{:<8} {:<6} {:>10}  {:<32} {:>12} {:>12.2} {:>10}", r.ticker, together::kind(r.etf), pct(r.vol), corr_words(r.corr), before(r.corr_before), r.ratio, pct(r.risk_left));
    }
}

pub fn hedge(ticker: &str, show: usize, as_of: &Option<String>) -> Result<()> {
    let u = universe(as_of, 2)?;
    let r = together::hedges(&u, ticker, show)?;
    println!("Hedge finder for {} ({}, {} a year volatility). Daily returns {}; a year earlier: {}.", r.ticker, together::kind(r.etf), pct(r.vol), span(r.days), span2(r.before_days));
    println!("Hedge ratio: dollars of it per $1 of {} that offset the most risk (sell it when the ratio is positive, buy it when negative). Risk left: the share of {}'s volatility that remains.", r.ticker, r.ticker);
    hedge_table(&format!("Tracks {} most closely:", r.ticker), &r.closest);
    hedge_table(&format!("Moves most opposite to {}:", r.ticker), &r.opposite);
    println!("\nLooked at {} liquid stocks and funds. Leveraged and inverse funds are included; their volatility column shows which they are.", r.scanned);
    Ok(())
}

pub fn groups(k: usize, stocks: usize, years_back: usize, members: usize, as_of: &Option<String>) -> Result<()> {
    let u = universe(as_of, years_back + 1)?;
    let t0 = std::time::Instant::now();
    let m = together::groups(&u, k, stocks, years_back, &|msg| eprintln!("  {msg}"))?;
    println!(
        "Groups map: the {} most-traded stocks in {k} groups that move together once the market (the equal-weight average of these stocks, times each stock's beta) is taken out. Daily returns {}.",
        m.stocks.len(),
        span(m.days)
    );
    let earlier: Vec<String> = m.earlier.iter().map(|&(a, b)| format!("{}..{}", c2_data::date_string(a), c2_data::date_string(b))).collect();
    println!("Kept together: of each group's pairs of stocks, the share also grouped together when the same grouping is made on an earlier year alone ({}).", earlier.join(", "));
    for (g, grp) in m.groups.iter().enumerate() {
        let tracks = grp.tracks.as_ref().map(|(t, r)| format!("tracks {t} {r:+.2}")).unwrap_or_else(|| "tracks no fund".into());
        let kept: Vec<String> = grp.kept.iter().map(|k| k.map(|(s, c)| format!("{:.0}% (chance {:.0}%)", 100.0 * s, 100.0 * c)).unwrap_or_else(|| "-".into())).collect();
        println!(
            "\nGroup {} ({} stocks): {tracks}; together {:+.2} without the market, {:+.2} with it; the year {}{}",
            g + 1,
            grp.members.len(),
            grp.corr_relative,
            grp.corr_raw,
            if grp.year_return >= 0.0 { format!("+{}", pct(grp.year_return)) } else { pct(grp.year_return) },
            if kept.is_empty() { String::new() } else { format!("; kept together {} year(s) back: {}", kept.len(), kept.join(", ")) }
        );
        let names: Vec<&str> = grp.members.iter().take(members).map(|&i| m.tickers[i].as_str()).collect();
        println!("  {}{}", names.join(" "), if grp.members.len() > members { format!(" … and {} more", grp.members.len() - members) } else { String::new() });
    }
    println!("\n[{:.1}s]", t0.elapsed().as_secs_f64());
    Ok(())
}
