# Influence functions, bootstraps, and why we cluster by site

*A ground-up explanation of how `peatfire.modeling.did` computes standard errors,
and why `cluster_by="site"` is the default.*

This document answers four questions, in order:

1. What **is** the influence function ψ? (Built from a sample mean upward, with
   numbers you can check by hand.)
2. How do you get from ψ to a standard error **of the ATT**?
3. What is a **multiplier bootstrap**, and what does it have to do with clustering?
4. Why does asking for site clustering **force** a bootstrap?

Every number below is real output from a short script; the code that produced each
block is shown with it, so you can re-run any of it.

---

## The one-paragraph version

Every estimator we use can be rewritten as *an average of one number per pixel*.
That number is the influence function ψᵢ — pixel *i*'s leverage on the answer. Once
you see the estimator as `mean(ψ)`, computing its standard error is just "the
standard error of a mean," and the **only** question that matters is whether the
terms in that average are independent. Pixels within a restoration site are not
independent, so the average has fewer independent terms than it appears to, and
treating each pixel as its own term makes the SE too small.

---

## Part 1 — ψ for a sample mean

Start with the simplest estimator there is. Six numbers, and θ̂ is their mean.

```python
x = np.array([2.0, 5.0, 4.0, 9.0, 3.0, 7.0])
theta = x.mean()
psi   = x - theta          # <- this is the influence function
```

```
x        = [2. 5. 4. 9. 3. 7.]
theta^   = mean(x) = 5.0000
psi      = x - theta^ = [-3.  0. -1.  4. -2.  2.]
```

That's it. **For a sample mean, ψᵢ is just the deviation of observation i from the
mean.** No mystery. Note two properties, which hold for *every* influence function:

```
property 1: mean(psi)   = 0.0000000000   (zero by construction)
property 2: sd(psi)/sqrt(n) = 1.064581
            textbook s/sqrt(n) = 1.064581   <- same thing
```

Property 2 is the whole game: **the standard error of the estimator is the standard
deviation of ψ, divided by √n.** For the sample mean this is the `s/√n` you already
know — the influence-function machinery just reproduces it.

### What "influence" actually means

Here's the interpretation that makes ψ concrete. ψᵢ tells you **how much the
estimate would move if you deleted observation i**:

$$\hat\theta_{(-i)} - \hat\theta \;=\; -\frac{\psi_i}{n-1}$$

Verified on the six numbers above — drop each one, recompute the mean, and rescale:

```
 i    x_i  theta^_(-i)     shift   -(n-1)*shift    psi_i
 0    2.0       5.6000    0.6000        -3.0000  -3.0000
 1    5.0       5.0000    0.0000        -0.0000   0.0000
 2    4.0       5.2000    0.2000        -1.0000  -1.0000
 3    9.0       4.2000   -0.8000         4.0000   4.0000
 4    3.0       5.4000    0.4000        -2.0000  -2.0000
 5    7.0       4.6000   -0.4000         2.0000   2.0000
```

The last two columns match exactly. Observation 1 (`x=5`, dead on the mean) has
ψ = 0: **deleting it changes nothing, so it has no influence.** Observation 3
(`x=9`, the outlier) has the largest ψ: deleting it moves the estimate most.

> **ψᵢ = pixel i's leverage on the answer.** Big |ψ| = this observation is
> driving the result. ψ = 0 = the estimate doesn't care whether it exists.

That is the entire concept. Everything below is the same idea applied to
progressively fancier estimators.

---

## Part 2 — ψ for a difference in means

Now θ̂ = (mean of treated) − (mean of controls). ψ has to do two things: give
treated and control pixels **opposite signs**, and **scale by group size** (if
only 10% of your pixels are treated, each treated pixel carries ten times the
leverage).

```python
p  = d.treated.mean()                       # share treated
d["psi"] = np.where(d.treated == 1,
                    (d.y - y_treated_mean) / p,        # treated: positive side
                   -(d.y - y_control_mean) / (1 - p))  # control: negative side
```

```
 pixel  treated   y  psi
     1        1 3.0 -4.0
     2        1 5.0  0.0
     3        1 4.0 -2.0
     4        1 8.0  6.0
     5        0 2.0 -0.0
     6        0 1.0  2.0
     7        0 4.0 -4.0
     8        0 1.0  2.0

mean treated = 5.0000, mean control = 2.0000, theta^ = 3.0000
share treated p = 0.50
mean(psi) = 0.0000000000
SE = sqrt(sum(psi^2))/n     = 1.118034
textbook sqrt(v_t/4+v_c/4) = 1.118034   <- same thing
```

Again ψ reproduces the textbook two-sample standard error exactly. Pixel 4
(`y=8`, far above the treated mean) has the biggest leverage; pixel 2 sits on the
treated mean and has none.

*(From here on I use `SE = sqrt(sum(psi²))/n`, which is the same as `sd(ψ)/√n`
with the plug-in variance convention — the form the code actually uses.)*

---

## Part 3 — ψ for a 2×2 DiD

Same structure, but each pixel is first collapsed to its **change** `Δ = y_post −
y_pre`, and then we difference treated against control.

```
 pixel  treated  y_pre  y_post  delta  psi
     1        1    0.3     0.1   -0.2  0.0
     2        1    0.2     0.1   -0.1  0.2
     3        1    0.4     0.2   -0.2  0.0
     4        1    0.3     0.0   -0.3 -0.2
     5        0    0.2     0.2    0.0  0.0
     6        0    0.3     0.4    0.1 -0.2
     7        0    0.1     0.1    0.0  0.0
     8        0    0.2     0.1   -0.1  0.2

treated mean change = -0.2000
control mean change = +0.0000
ATT = -0.2000 - (+0.0000) = -0.2000
mean(psi) = 0.0000000000
SE = sqrt(sum(psi^2))/n = 0.050000
```

Read the ψ column as "how unusual was this pixel's *change*, relative to its own
group's average change." Pixels 1 and 3 changed by exactly the treated average
(−0.2), so ψ = 0. Pixel 4 fell more than average (−0.3) and pixel 2 less (−0.1),
so they pull the ATT in opposite directions.

**This is already the DiD you'd compute by hand.** The Callaway–Sant'Anna
estimator is this, once per (cohort, year) cell, with covariate adjustment.

---

## Part 4 — ψ for the *real* doubly-robust estimator

The estimator in `estimate_att` adds two things: an **outcome regression** m̂(X)
predicting a control pixel's change from its covariates, and a **propensity score**
ê(X) weighting controls by how much they resemble treated pixels. Schematically:

$$\widehat{ATT}(g,t) = \frac{1}{n}\sum_i \big(\underbrace{w^{\text{treat}}_i - w^{\text{ctrl}}_i}_{\text{weight from } \hat e}\big)\big(\underbrace{\Delta Y_i - \hat m(X_i)}_{\text{residual}}\big)$$

and ψᵢ is pixel *i*'s term in that sum, centered, **plus correction terms for the
fact that ê and m̂ were themselves estimated from the data.**

You do not have to take the algebra on faith. If ψ really means "leverage," then
the leave-one-out check from Part 1 should still work — even though this estimator
fits two nuisance models internally. So: refit the **entire** doubly-robust
estimator 40 times, dropping one pixel each time, and compare.

```python
model, ntl = fit_att_gt(small)              # 40 pixels x 4 years
psi = stack_influence_funcs(ntl)            # the stored influence functions

jack = [fit_att_gt(small.drop(index=u, level=0))[1] ... for u in units]
implied = -(n - 1) * (jack - base_att)      # jackknife estimate of psi
```

```
panel: 40 pixels x 4 years
psi shape: (40, 3)  -> (pixels, (g,t) cells)
using cell (2020, 2019, 2021), ATT = -0.10000

 pixel  treated  ATT without it   -(n-1)*shift     psi_i
     1     True        -0.11316        0.51316   0.50000
     2     True        -0.16579        2.56579   2.50000
     3     True        -0.11316        0.51316   0.50000
     4     True        -0.06053       -1.53947  -1.50000
    21    False        -0.03947       -2.36053  -2.30000
    22    False        -0.09211       -0.30789  -0.30000
    23    False        -0.14474        1.74474   1.70000
    24    False        -0.14474        1.74474   1.70000

correlation(jackknife-implied psi, stored psi) = 1.00000
mean abs difference = 0.03342
```

**Correlation 1.00000.** The stored influence functions are, to a first-order
approximation, exactly "how much does the ATT move if I drop this pixel." (The
small absolute gap is because the jackknife is a first-order approximation, not
because ψ means something different.)

And the same two properties still hold:

```
mean(psi) = 6.661e-17
sd(psi)/sqrt(n) = 0.236114   reported std_error = 0.236114
```

That second line is the answer to **"how do we get the SE of the ATT rather than
of the propensity score and outcome regression?"**

> ψ is built by differentiating the **ATT estimator itself** — the whole pipeline,
> nuisance models included. The chain rule through ê and m̂ is exactly what
> produces those correction terms. ê and m̂ never get standard errors of their own;
> their sampling variability enters *only* as extra spread in ψ. There is one
> estimator, one influence function, one standard error, and it belongs to the ATT.

At the scale of a real run, ψ is a matrix — one row per pixel, one column per (g,t)
cell. From a larger simulated panel (720 pixels, 6 sites, 12 years, 4 staggered
cohorts — the same shape as your matched panel, just smaller):

```
psi shape: (720, 66) = (pixels, (g,t) cells)
exactly zero: 42% of pixels   <- pixels this comparison doesn't use
```

Pixels outside a given comparison have ψ = 0 for that column: zero leverage, exactly
as the definition says. The overall ATT and event study are weighted sums of the
columns, so their influence functions are the same weighted sums — which is why
`aggregate_att` has to rebuild ψ and be told the clustering separately.

---

## Part 5 — From ψ to a standard error: two roads

We now know `θ̂ − θ ≈ mean(ψ)`. Getting a standard error means asking: **how much
does `mean(ψ)` bounce around?** There are two ways to answer.

### Road A — the closed form ("analytic")

Evaluate a formula, once:

```python
# differences/models/attgt/utility_ntl.py:458
vcv = (inf_funcs.T @ inf_funcs) / n
return np.sqrt(np.diag(vcv) / n)          # = sd(psi)/sqrt(n)
```

Instant, deterministic. **But** going from `Var(mean ψ)` to `(1/n²)·Σψᵢ²` requires
the ψᵢ to be **independent across rows**. That assumption is baked into the algebra
before you ever call the function.

### Road B — the multiplier bootstrap

An ordinary bootstrap resamples rows and **refits the model** 1000 times. For this
estimator that's 1000 doubly-robust fits per cell — impractical.

The multiplier bootstrap never resamples and never refits. **The models are fit
exactly once.** ψ is computed once and frozen. Then you draw random ±1 weights and
*multiply* them into the frozen ψ — hence "multiplier":

$$\theta^*_b = \frac{1}{n}\sum_i V_i\,\psi_i, \qquad V_i = \pm 1 \text{ with prob } \tfrac12$$

Six pixels, six draws, by hand:

```
psi = [ 2. -1.  3. -4.  1. -1.]

 draw  signs                      V*psi                              mean
    1  [ 1  1 -1  1 -1 -1]        [ 2. -1. -3. -4. -1.  1.]       -1.0000
    2  [-1  1  1  1  1 -1]        [-2. -1.  3. -4.  1.  1.]       -0.3333
    3  [-1  1 -1  1 -1 -1]        [-2. -1. -3. -4. -1.  1.]       -1.6667
    4  [-1 -1 -1  1  1 -1]        [-2.  1. -3. -4.  1.  1.]       -1.0000
    5  [ 1 -1 -1  1 -1  1]        [ 2.  1. -3. -4. -1. -1.]       -1.0000
    6  [ 1 -1  1 -1  1 -1]        [2. 1. 3. 4. 1. 1.]              2.0000

spread over 200,000 draws: sd = 0.943992
closed form  sqrt(sum(psi^2))/n = 0.942809
```

**The two roads agree.** They are the same computation — one done algebraically,
one by simulation. In `differences` this is `mboot.py:137`:

```python
ub = rng.choice([1, -1], size=(n, 1))     # one sign per ROW of psi
out_mat[bit] = np.mean(inf_funcs * ub, axis=0)
```

So why bother with Road B if it just reproduces Road A? Because of what you can
change about it — which is Part 6.

---

## Part 6 — Clustering: the site averages of ψ

Here is the pivot. In `mean(ψ)`, the noise cancels out *because* the ψᵢ are
independent and mean-zero. If they're **not** independent, less cancels than the
formula assumes, and the SE comes out too small.

The diagnostic is simple: **average ψ within each site.** If pixels inside a site
are independent, those averages should be near zero (positives and negatives
cancel). If they're not, the averages survive.

12 pixels, 3 sites of 4, where every pixel in a site moves together:

```
  psi        = [ 2.5  2.5  2.5  2.5 -3.  -3.  -3.  -3.   0.5  0.5  0.5  0.5]
  site means = [ 2.5 -3.   0.5]
  SE, pixels independent = 0.6562
  SE, sites  independent = 1.3123     ratio = 2.00x
```

Nothing cancels within a site — the site averages are as large as the individual
pixels. The pixel-level formula divides by 12 when there are really only **3**
independent things, and it's wrong by exactly √4 = 2, the square root of the
pixels per site.

Both SEs are the same shape — "root sum of squares, divided by how many
independent things you have":

| | formula | independent units |
|---|---|---|
| pixel-clustered | `sqrt(Σᵢ ψᵢ²) / n` | n pixels |
| site-clustered | `sqrt(Σₛ ψ̄ₛ²) / G` | G sites |

**And that is all clustering is:** collapse ψ to one row per site by averaging,
then do the same arithmetic on G rows instead of n. In `differences` that is
`mean_inf_func_by_cluster` (`mboot.py:30`), called right before the sign-flipping.

Which makes the bootstrap connection obvious:

- one ±1 **per pixel** → pixels flip independently → pixel-level SE
- one ±1 **per site**, applied to all pixels in it → whole sites flip together → site-clustered SE

**The unit you assign a random weight to is the unit you are treating as
independent.**

This is not just a toy property. Here are the site averages of ψ from the 720-pixel
fitted model above, where the simulation deliberately gave each site a shared
year-by-year shock:

```
site means of psi: [ 0.0333  0.15  0.1333 -0.0333 -0.1333 -0.15 ]
```

Each of those averages 120 pixels. If pixels within a site were independent,
averaging 120 mean-zero numbers would land near 0. These are systematically ±0.15.
**That non-vanishing average is the within-site correlation** — the thing the
pixel-level formula silently throws away.

---

## Part 7 — How big is the effect? The design effect

The inflation factor has a name and a formula:

$$\text{design effect} = \sqrt{1 + (m-1)\rho}$$

where *m* = pixels per site and ρ = intra-site correlation. Simulated check
(6 sites, ψ = site component + pixel noise):

```
 pixels/site     rho   SE pixel    SE site   observed   sqrt(1+(m-1)rho)
          10    0.00    0.12813    0.11667      0.91x              1.00
          10    0.01    0.12762    0.12354      0.97x              1.04
          10    0.10    0.12714    0.16310      1.28x              1.38
         100    0.00    0.04078    0.03749      0.92x              1.00
         100    0.01    0.04076    0.05266      1.29x              1.41
         100    0.10    0.04046    0.12283      3.04x              3.30
        1000    0.00    0.01291    0.01189      0.92x              1.00
        1000    0.01    0.01290    0.03911      3.03x              3.32
        1000    0.10    0.01279    0.11806      9.23x             10.04
```

Two things to read here.

**First, the ρ column is doing the work, but *m* is the amplifier.** Note the
`SE pixel` column keeps shrinking as m grows (0.128 → 0.041 → 0.013) *regardless of
ρ* — that's the 1/√n behaviour that makes pixel-level SEs look so good. The site
column doesn't shrink once ρ > 0.

This matters for your study, because *m* is large. The matched run reports 3744
matched pairs over 6 sites — so ~624 treated pixels per site, or ~1248 panel pixels
per site once the matched controls are counted:

```
       rho      m=624     m=1248
    0.0001      1.03x      1.06x
     0.001      1.27x      1.50x
      0.01      2.69x      3.67x
      0.05      5.67x      7.96x
```

An intra-site correlation of **one percent** — pixels in the same peatland being 1%
alike, which is a very modest claim about fire behaviour — already inflates the
honest SE by roughly 3×. You don't need dramatic within-site correlation for this
to matter; you need *any*, because *m* multiplies it.

**Second, look at the ρ = 0.00 rows: 0.91–0.92×, not 1.00×.** That is not noise.
With G = 6 sites, the uncorrected site SE is biased low by exactly √(5/6) = 0.913.
This is why `att_collapsed` applies the `G/(G−1)` finite-cluster correction and
reports a `t(G−1)` interval rather than a normal one — with 6 clusters those
corrections are not pedantry, they're a 10%+ effect on their own.

---

## Part 8 — Why site clustering *forces* a bootstrap

Now the punchline, which should follow for free.

Clustering lives **entirely** in how the random weights are drawn: one per pixel,
or one per site. The closed form has **no random weights at all** — it's a single
matrix product. There is no slot in `(ψᵀψ)/n²` to say "rows 1–3700 all belong to
site0."

This isn't a missing feature. The closed-form function takes no cluster argument
because **the derivation assumed independence** before the formula was written
down. `differences` and R's `did` both work this way: `cluster_groups` is only
consulted on the `boot_iterations > 0` path (`attgt_cal.py:690`).

So:

> **"Closed-form SE" and "pixel-level SE" are the same thing here.** Asking for
> site clustering means asking for the bootstrap, because the bootstrap is where
> the clustering physically happens.

That's the cost of `cluster_by="site"`: a 1000-draw bootstrap instead of one matrix
product. No refitting — just 1000 sign-flips of a frozen matrix.

### One caveat specific to your G

With 6 sites there are only 2⁶ = 64 distinct sign patterns. The bootstrap
distribution is genuinely coarse no matter how many draws you request, so p-values
near conventional thresholds should not be read precisely. This is a property of
having 6 restoration sites, not of the software.

---

## Part 9 — What this means in `did.py`

| Step | Where | What it does |
|---|---|---|
| Make the key | `prepare_panel` (`did.py:264`) | Materializes `site_id`; unmatched controls (which belong to no site) get singleton clusters + a warning |
| Carry the key | `build_panel` (`did.py:424`) | Keeps `site_id` on the panel — it used to be dropped — and rejects a time-varying or null key |
| Resolve the toggle | `_resolve_cluster` (`did.py:521`) | `"site"` → column name, `"pixel"` → `None`; warns below 30 clusters |
| Fit + choose the road | `estimate_att` (`did.py:603`) | `boot_iterations` defaults to 1000 under `"site"`, 0 under `"pixel"`; `0` **with** site clustering raises rather than silently downgrading |
| Work around upstream | `_attach_cluster_column` (`did.py:567`) | `differences` 0.3.0's own `fit(cluster_var=...)` is broken; this does by hand the one line it should have done |
| Cluster the *reported* numbers | `aggregate_att` (`did.py:763`) | Inherits the fit's clustering. Easy to miss and critical: `differences` **re-derives** its SEs during aggregation, so without this the headline ATT reverts to pixel-level SEs |
| Cross-check by hand | `att_collapsed` (`did.py:899`) | One pre/post change per pixel → one θₛ per site → SE from their spread, `G/(G−1)` corrected, `t(G−1)` interval. No optional dependencies |

Usage:

```python
att, overall, es = did.fit_att(panel, covariates=covs)                     # site (default)
att_p, overall_p, es_p = did.fit_att(panel, covariates=covs, cluster_by="pixel")  # the deflated one

# Both runs in one table: same att, two std_errors, design_effect = se_site/se_pixel.
did.compare_cluster_levels({"site": overall, "pixel": overall_p})
did.compare_cluster_levels({"site": es, "pixel": es_p})   # per event time, too

r = did.att_collapsed(panel)          # transparent cross-check
r["by_site"]                          # the per-site theta_s table
r["design_effect"]                    # se_site / se_pixel, the number from Part 7
```

This is what `notebooks/modeling.ipynb` now does at both DiD stages (§2b on the
unmatched panel, §2c on the matched one), plus the `att_collapsed` cross-check on
the matched panel — the one site-clustered ATT in the notebook that needs no
optional backend.

**Quick check on which SEs you're looking at:** the `std_error` column header reads
`bootstrap` when clustered and `analytic` when not.

---

## Cheat sheet

| Term | Meaning |
|---|---|
| **ψ (influence function)** | One number per pixel: how much the estimate moves if you drop that pixel (×n). Mean zero by construction. |
| **`θ̂ − θ ≈ mean(ψ)`** | Every estimator here is an average of ψ's. So every SE is "the SE of a mean." |
| **Closed-form / analytic SE** | `sd(ψ)/√n`, evaluated once. Assumes ψᵢ independent → equivalent to pixel-level clustering. |
| **Multiplier bootstrap** | Multiply frozen ψ by random ±1 weights, 1000×, measure the spread. No refitting. |
| **Clustering** | Average ψ within each site first, then treat the G site-averages as the independent units. |
| **Design effect** | `√(1 + (m−1)ρ)`. With m ≈ 3700 pixels/site, even ρ = 0.01 gives 6×. |
| **Why bootstrap is forced** | Clustering lives in how the weights are drawn; the closed form has no weights. |

## Further reading

- Callaway, B. & Sant'Anna, P. H. C. (2021). *Difference-in-Differences with
  Multiple Time Periods.* Journal of Econometrics 225(2). — the estimator, its
  influence functions, and the multiplier bootstrap.
- Sant'Anna, P. H. C. & Zhao, J. (2020). *Doubly Robust Difference-in-Differences
  Estimators.* Journal of Econometrics 219(1). — the DR ATT(g,t) and its ψ.
- Bertrand, M., Duflo, E. & Mullainathan, S. (2004). *How Much Should We Trust
  Differences-in-Differences Estimates?* QJE 119(1). — the paper that made
  clustering standard practice in DiD; `att_collapsed` implements its
  collapse-to-one-observation-per-cluster remedy.
- Cameron, A. C. & Miller, D. L. (2015). *A Practitioner's Guide to Cluster-Robust
  Inference.* Journal of Human Resources 50(2). — the few-clusters problem, i.e.
  the G ≈ 6 caveat that runs through this whole document.
