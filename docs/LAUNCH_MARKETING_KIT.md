# NeuralTrend Launch & Marketing Kit

Everything here is written to stay consistent with the legal positioning in
`LAUNCH_READINESS_PASS2.md` — informational research tool, not personalized
advice, no performance guarantees. Keep that discipline in anything you
adapt below; it's not just a legal nicety, it's the thing that keeps you
inside the investment-adviser publisher's exclusion.

## 1. Positioning

**One-line pitch:** NeuralTrend is a research platform that turns an AI model's
market read into a transparent, backtested BUY/HOLD/SELL signal for crypto and
stocks — with the full history, methodology, and track record public before
you ever pay for anything.

**What makes this different from "another crypto signals Telegram":**
- Public, dated Methodology and Performance pages — most signal sellers show
  you nothing until you pay.
- A Forward Record that's locked in before the fact (you can't retroactively
  edit a published call), which is the single biggest trust gap in this
  category.
- No DMs, no "VIP group," no urgency countdown timers — subscription SaaS
  pricing, cancel anytime via Stripe's own portal.

Lead with **transparency and track record**, not with predictions of what the
market will do next. The second one is both legally risky and, frankly, the
same pitch every low-trust signals account already makes — it's not your
differentiator. Transparency is.

## 2. Who you're actually selling to

Three overlapping groups, in likely order of conversion ease:

1. **Self-directed crypto/stock traders who already use TradingView,
   backtesting tools, or spreadsheets** — they understand what a backtest is
   and will actually read your Methodology page. Highest-intent, easiest to
   convert, but smallest pool.
2. **"Quant-curious" retail investors** who follow fintwit/crypto-Twitter,
   listen to markets podcasts, and want a research edge but don't want to
   build a model themselves.
3. **r/algotrading and adjacent developer-investor communities** — they will
   stress-test your methodology publicly. Treat this as a feature: if your
   Forward Record holds up under scrutiny there, it's your best word-of-mouth
   engine.

## 3. Channel reality check — read this before budgeting ad spend

I checked current (2026) ad policy on the two platforms founders default to,
because this genuinely changes the plan:

- **Google Ads: as of Google's current policy, "cryptocurrency investment
  advice and trading signal platforms" are explicitly prohibited outright —
  regardless of certification.** This isn't a "get verified and you're fine"
  situation like it is for exchanges/wallets; it's a named, categorical
  block. Your crypto-signal product (which is the flagship/free tier) is very
  likely to be rejected here no matter how the ad is worded. The stock-signal
  side is a slightly different policy bucket ("complex speculative financial
  products" / trading-platform certification, which requires proof of
  regulatory registration you don't have by design), so don't count on Google
  Search/Display for this business at launch.
- **Meta (Facebook/Instagram): crypto ads are allowed but gated.** Purely
  educational/informational crypto content can run without special
  permission; anything Meta reads as transactional/investment-related
  (which a paid buy/sell/hold signal product plausibly is) needs written
  authorization plus proof of a regulatory license — which again, you don't
  have by design. Realistic path: run ads for the *free educational content*
  (Methodology, "how AI signals work") without ROI/performance claims, funnel
  to the site, let the product convert organically from there — not a direct
  "subscribe now" performance ad.
- **X/TikTok/other platforms** have their own, similarly-restrictive
  financial/crypto ad policies. Check each one's current policy immediately
  before spending anything — these change often enough that whatever I'd tell
  you here could be stale within months.

**Bottom line:** don't build the launch plan around paid social/search ads.
Budget them as a maybe-later experiment with realistic expectations, and put
the real effort into organic/community/content channels below, which have no
such restriction and, for this specific audience, will likely outperform ads
anyway (this audience is unusually ad-skeptical and unusually willing to
click through to a methodology page).

## 4. Channels to actually use

- **Reddit** (r/algotrading, r/CryptoCurrency, r/CryptoMarkets,
  r/quant, r/stocks — check each subreddit's current self-promotion rule
  before posting, they vary and change). Reddit's own guidance is the 90/10
  rule: be a redditor with a website, not a website with a Reddit account.
  Concretely: spend 2–3 weeks genuinely participating (answering questions,
  no links) before your first NeuralTrend mention, disclose that you're the
  founder whenever you post about it, never post the identical copy to
  multiple subreddits same-day, and lead with the Methodology/Forward Record
  page rather than the pricing page.
- **Product Hunt** — this audience (builders, quant-curious, fintech-adjacent)
  is close to your ICP, and "public, dated track record" is a strong PH
  narrative. Launch on a Tuesday–Thursday, have your Methodology/Performance
  pages ready to link from the first comment.
- **Hacker News** ("Show HN") — works well if you frame it around the
  engineering/methodology (backtesting approach, forward-record integrity
  design) rather than as a product pitch. HN punishes anything that reads as
  marketing.
- **X/Twitter (organic)** — fintwit and crypto-Twitter both reward
  "here's the data" threads. A weekly "here's what the signal said and what
  actually happened" recap thread is genuinely good content and a natural
  place to link the live Performance page.
- **LinkedIn** — underrated for this specifically because "LLC, transparent
  methodology, no advice, no custody" is a credibility story that resonates
  with a more risk-aware/professional audience than crypto-Twitter.
- **SEO / content** — your Methodology, Performance, and comparison pages
  (`buy_and_hold_vs_ai_strategy.html`, `ai_crypto_trading_signals.html`) are
  already built for this. Each supported ticker is a plausible
  "[Ticker] AI signal" long-tail page/post if you want to extend this later.
- **Newsletter/creator sponsorships** — a different regime than platform ads
  (you're paying a person, not buying a platform placement), but the creator
  still needs to disclose the paid relationship (FTC endorsement rules apply
  regardless of platform), and you should still avoid putting performance
  claims in their read — give them the Methodology/Performance link and let
  them draw their own (accurate) conclusions.

## 5. Launch sequence

**Pre-launch (this week):**
1. Close out the two legal items in `LAUNCH_READINESS_PASS2.md` — or at
   minimum, get a preliminary read from counsel before you drive real traffic
   and real subscriptions.
2. Run the Pass 1 staging checklist end to end.
3. Set up basic analytics you can actually check post-launch: Stripe
   dashboard for conversions, Simple Analytics for traffic, and Admin
   Operations for system health.
4. Prep 3–5 pieces of content in advance (see samples below) so launch day
   isn't spent writing from scratch.

**Launch day:**
1. Product Hunt / Show HN post (pick one primary platform to launch on, not
   both same day — spreads attention thin).
2. Your own X/LinkedIn announcement, linking straight to the Methodology or
   Performance page, not just the homepage.
3. Reply personally to every early comment/question — at this stage, you
   *are* the trust signal.

**Week 1–4:**
1. One "signal recap" post per week (what the model said, what happened,
   link to live Performance page) — this is your core recurring content
   format and it writes itself once the Forward Record has a few weeks in it.
2. Answer questions in 2–3 relevant subreddits/communities without linking,
   to build the account history the 90/10 rule assumes.
3. Watch Stripe + Admin Operations daily for the first two weeks; this is
   when webhook/edge-case issues surface under real traffic.

## 6. Ready-to-use drafts

### X / Twitter — launch post
> NeuralTrend is live: AI-generated BUY/HOLD/SELL signals for BTC, ETH, SOL,
> XRP + a set of stocks — with a public Methodology page and a Forward Record
> that's locked in before the fact, not edited after.
>
> Not financial advice. Just the model, the history, and the receipts.
> [link]

### X / Twitter — weekly recap format (reusable template)
> Week of [date] — NeuralTrend Signal Recap
>
> BTC: signal was [BUY/HOLD/SELL] on [date]. Since then: [+/-X%].
> ETH: ...
>
> Full history + methodology: [link to Performance page]
> Research tool, not advice — do your own diligence.

### LinkedIn — launch post
> After [X months] of building, NeuralTrend is public: a research platform
> that publishes AI-generated market signals for crypto and stocks with a
> transparent, dated track record.
>
> The premise is simple: most "AI trading signal" products ask you to trust
> them. We'd rather show you — every signal is published with its
> methodology and its actual forward performance, and once a date is
> published it isn't quietly revised later.
>
> It's a research and education tool, not personalized financial advice —
> and it's built that way on purpose. [link]

### Reddit post skeleton (adapt per subreddit, check rules first)
> Title: Built a transparent AI signal tracker with a locked-in forward
> record — feedback welcome
>
> Body: I've been building NeuralTrend, [one line on the model/approach].
> The part I care most about: the Methodology page explains exactly how
> signals are generated, and the Performance page shows the actual forward
> record — dates are locked once published, nothing gets revised after the
> fact. Free tier covers BTC/ETH/SOL/XRP. Genuinely interested in this
> community's take on the methodology, not just the pitch — happy to answer
> anything about how it works. [link to Methodology, not the pricing page]

### Product Hunt tagline options
- "AI market signals with a public, tamper-evident track record."
- "See the signal, see the history, then decide — not financial advice."

### Meta ad copy (educational-tier safe example, if you pursue this later)
> Curious how an AI model reads BTC/ETH price action? See NeuralTrend's public
> methodology and historical signal record — free to explore, no account
> needed. [link to Methodology page]
(No ROI/return numbers, no "buy now," no urgency language — this is what
keeps it in Meta's lower-scrutiny educational bucket. Expect it to still need
their crypto-ads review process.)

## 7. Guardrails for anything you write beyond this doc

- Never state or imply a specific future return, win rate, or "you'll make
  money." Past-performance language stays in the past tense and links to the
  actual record.
- Don't use "advice," "recommend," "you should buy/sell," or second-person
  imperative language about specific trades — that's the exact phrasing the
  ToS/Risk Disclaimer say you don't do.
- Disclose paid relationships (creator sponsorships, any affiliate
  arrangement) — this is an FTC requirement independent of platform policy.
- If a number appears in an ad or post (a return %, an outperformance ratio),
  it should trace directly to something on the public Performance page — never
  a cherry-picked backtest window that isn't the one visitors will land on.

## 8. Content calendar

Laid out as relative weeks so you can map it onto your actual launch date.
"Warm-up" must start before launch — the whole point is that your Reddit/X
accounts don't look brand-new the day you start posting links.

### Warm-up (the 1–2 weeks before launch)
| Day | Platform | Action |
|---|---|---|
| W-14 to W-1 | Reddit | Create/use a real account with a bio. Comment genuinely in r/algotrading, r/quant, r/CryptoCurrency, r/stocks — answer questions, zero links. This is the step people skip and it's the one that determines whether your launch post survives. |
| W-7 | X/LinkedIn | Start posting build-in-public style updates (no link needed yet) — "shipping the Forward Record integrity check this week," etc. |
| W-3 | Email/list | If you have any waitlist/existing contacts, send a "we're launching in ~3 weeks" heads-up. |
| W-1 | All | Prep launch-day assets: PH/HN post, X thread, LinkedIn post, screenshots of the Methodology/Performance pages. Draft, don't post yet. |

### Week 1 (launch week)
| Day | Platform | Content |
|---|---|---|
| Mon | Product Hunt *or* Show HN (pick one) | Launch post. Reply to every comment same day. |
| Mon | X | Launch thread (template above). |
| Mon | LinkedIn | Launch post (template above). |
| Wed | X | "How the signal actually works" thread — walk through Methodology in plain language, 1 concept per tweet. |
| Fri | Reddit | First on-topic post in the subreddit where your warm-up has been strongest (see §9) — lead with methodology, not the product. |

### Week 2
| Day | Platform | Content |
|---|---|---|
| Mon | X + LinkedIn | First weekly Signal Recap (template above) — this becomes your permanent Monday slot. |
| Wed | Blog/site or X | "Why we lock the Forward Record before the fact" — the integrity/trust angle, your strongest differentiator. |
| Fri | Reddit | Second subreddit, different community than Week 1. |

### Week 3
| Day | Platform | Content |
|---|---|---|
| Mon | X + LinkedIn | Weekly Signal Recap. |
| Wed | X | Ticker deep-dive — pick one supported asset, show its full signal history and what it would have meant for a $10k position. Pull the actual numbers from the Performance page. |
| Fri | Press | Send 3–5 pitch emails (template in §10). |

### Week 4
| Day | Platform | Content |
|---|---|---|
| Mon | X + LinkedIn | Weekly Signal Recap. |
| Wed | Reddit or X | Respond publicly to the best piece of community feedback/criticism from weeks 1–3 — "you said X, here's what we changed/clarified." This is disproportionately good for trust in this audience. |
| Fri | Retro | Look at Stripe conversions + traffic by source (Simple Analytics) and double down on whichever channel actually converted, not whichever felt biggest. |

After week 4, the Monday Signal Recap + occasional ticker deep-dive is a
sustainable ongoing cadence on its own — you don't need a new content format
every week, you need the recurring one to compound.

## 9. Community-by-community notes

Being straight with you on this one: Reddit is one of the platforms where
generic third-party guides you find by searching are frequently stale or
wrong, and I don't have a reliable way to pull the live, current sidebar/wiki
rules for a specific subreddit from here. So treat everything below as
*reputation-level* guidance (which communities are historically strict about
vendor/self-promo, and how) rather than a guarantee of the current rule text
— **the one non-negotiable step is opening each subreddit's "rules" tab
yourself, same day you post, since rules and required flairs do change.**

- **r/algotrading** — historically one of the strictest about anything that
  reads as vendor/product promotion; posts get reported by the community
  itself, not just removed by mods. Only viable angle here is a genuinely
  technical post about the *engineering* (how you built the forward-record
  integrity system, how you handle backtest look-ahead bias) with the product
  link secondary, if allowed at all.
- **r/CryptoCurrency** — huge, very active, historically requires a minimum
  account "karma" threshold before you can post/comment at all, and has its
  own self-promotion rule that's stricter than Reddit's site-wide default.
  Worth it if you clear the bar, but budget real warm-up time.
- **r/CryptoMarkets** — smaller, generally more tolerant of trading-tool
  discussion than r/CryptoCurrency, still check the current self-promo rule.
- **r/quant** — skews academic/professional; a methodology-first post (how
  you validate signals, how you think about overfitting) will land far
  better than anything that reads as a pitch.
- **r/stocks** — large, general-audience; typically wary of anything
  resembling stock promotion/pumping given the subreddit's history with that
  exact abuse pattern, so lead with the free educational content, not a call
  to subscribe.
- **Easier entry points to practice on first:** r/SideProject, r/SaaS, and
  r/EntrepreneurRideAlong are explicitly founder/launch-friendly by culture
  (many have dedicated "show off your project" threads) — good places to get
  comfortable with the format and build some account history before you
  tackle the finance-specific subs above.

## 10. Press outreach

You don't have much of a track record on day one, so early outreach should
be light-touch and angle-driven rather than a hard "cover us" pitch. Angles
that fit what you've actually built:

- **The integrity angle**: "What it looks like when a signals product can't
  quietly rewrite its own track record after the fact" — pitch to fintech/
  regtech-adjacent newsletters.
- **The solo/small-team build angle**: the engineering-and-ops rigor
  (backups, admin MFA, idempotent Stripe webhooks, a documented ADR log) is
  genuinely unusual for a product this size — a good story for
  developer-and-founder-focused outlets, not just crypto press.
- **The "AI model, shown its homework" angle**: relevant to general AI-in-
  finance coverage, which is a hot beat right now — the differentiator is
  that you publish the methodology and the misses, not just the wins.

Research current crypto/fintech trade outlets and the specific reporters on
those beats yourself before pitching (that's the kind of fast-moving,
name-specific detail I'd rather you verify fresh than get from me) — but a
simple pitch template:

> Subject: A crypto/stock signals product that locks in its track record
> before the fact
>
> Hi [name] — following your coverage of [specific recent piece they wrote].
> I built NeuralTrend, an AI signal research platform for crypto and stocks
> with a public Methodology page and a Forward Record that's timestamped and
> can't be revised after publication — trying to fix the "trust me" problem
> most signal products have. Happy to walk you through the mechanics if
> useful for anything you're working on. [links to Methodology + Performance]

## 11. Short explainer script (homepage / Product Hunt video, ~40 seconds)

> Most trading-signal products ask you to trust them.
> NeuralTrend doesn't ask — it shows you.
>
> Every signal comes from a documented model, published with its methodology,
> and tracked in a Forward Record that's locked in the moment it's published
> — not editable after the fact.
>
> Free tier covers Bitcoin, Ethereum, Solana, and XRP. Pro unlocks the rest —
> stocks, full history, backtesting, and live paper simulations.
>
> It's research, not advice. See the model's actual track record before you
> ever pay for anything — that's the whole idea.
>
> NeuralTrend. See the signal. See the history. Then decide.
