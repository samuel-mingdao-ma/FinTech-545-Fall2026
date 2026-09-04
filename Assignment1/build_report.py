"""Build the written Assignment 1 submission as a polished PDF."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
RESULTS = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
PDF_PATH = ROOT / "assignment1.pdf"

NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#2A6F97")
TEAL = colors.HexColor("#2A9D8F")
ORANGE = colors.HexColor("#E76F51")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_SAND = colors.HexColor("#FCF4DF")
TEXT = colors.HexColor("#243447")
MUTED = colors.HexColor("#64748B")


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=14,
    )
)
styles.add(
    ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=BLUE,
        spaceAfter=7,
    )
)
styles.add(
    ParagraphStyle(
        name="Section",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=NAVY,
        spaceBefore=4,
        spaceAfter=9,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="Subsection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=BLUE,
        spaceBefore=7,
        spaceAfter=4,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.3,
        leading=13.3,
        textColor=TEXT,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=TEXT,
    )
)
styles.add(
    ParagraphStyle(
        name="SmallHeader",
        parent=styles["Small"],
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
)
styles.add(
    ParagraphStyle(
        name="Caption",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=7.8,
        leading=10,
        textColor=MUTED,
        alignment=TA_CENTER,
        spaceBefore=2,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="Callout",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13,
        textColor=TEXT,
        leftIndent=10,
        rightIndent=10,
        borderColor=TEAL,
        borderWidth=1,
        borderPadding=8,
        backColor=colors.HexColor("#ECF8F6"),
        spaceBefore=4,
        spaceAfter=8,
    )
)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def body(text: str) -> Paragraph:
    return Paragraph(text, styles["Body"])


def small(text: str) -> Paragraph:
    return Paragraph(text, styles["Small"])


def subsection(label: str, title: str) -> Paragraph:
    return Paragraph(f"{label}. {title}", styles["Subsection"])


def figure(filename: str, width: float, caption: str) -> list:
    image_path = OUTPUT / filename
    image = Image(str(image_path))
    image.drawHeight = image.imageHeight * width / image.imageWidth
    image.drawWidth = width
    return [image, Paragraph(caption, styles["Caption"])]


def styled_table(rows: list[list], widths: list[float], header: bool = True) -> Table:
    converted = []
    for row_index, row in enumerate(rows):
        cell_style = styles["SmallHeader"] if header and row_index == 0 else styles["Small"]
        converted.append(
            [
                cell if hasattr(cell, "wrap") else Paragraph(str(cell), cell_style)
                for cell in row
            ]
        )
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


def page_decoration(canvas, document) -> None:
    canvas.saveState()
    width, _ = letter
    canvas.setStrokeColor(colors.HexColor("#DCE4EC"))
    canvas.setLineWidth(0.5)
    canvas.line(0.72 * inch, 0.58 * inch, width - 0.72 * inch, 0.58 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.72 * inch, 0.38 * inch, "FinTech 545 - Assignment 1")
    canvas.drawRightString(width - 0.72 * inch, 0.38 * inch, f"Page {document.page}")
    canvas.restoreState()


def add_problem_1(story: list) -> None:
    r = RESULTS["problem1"]
    story.append(Paragraph("1. Reading the Shape of a Sample", styles["Section"]))
    story.append(subsection("Predict", "Moments and candidate families"))
    story.append(
        body(
            "Before fitting a distribution, I used the sample mean, unbiased sample variance, "
            "bias-corrected skewness, and bias-corrected excess kurtosis. The sample is clearly "
            "left-skewed and heavy-tailed. Therefore a general normal inverse Gaussian (NIG) "
            "distribution is plausible: it can represent both negative skew and positive excess "
            "kurtosis. A Normal is ruled out by both the nonzero skewness and positive excess "
            "kurtosis. A symmetric Student-t can represent the positive excess kurtosis but is "
            "ruled out by the negative skewness. A lognormal is ruled out by the sign of skewness: "
            "every nondegenerate lognormal has positive skew. This judgment uses the requested "
            "moments alone; the observed negative values would independently rule out a lognormal."
        )
    )
    rows = [
        ["Statistic", "Estimate", "Interpretation"],
        ["Mean", f"{r['mean']:.8f}", "Center is close to zero"],
        ["Sample variance", f"{r['variance']:.8f}", "Daily-scale dispersion"],
        ["Skewness", fmt(r["skewness"], 3), "Negative: longer/heavier left side"],
        ["Excess kurtosis", fmt(r["excess_kurtosis"], 3), "Positive: fatter tails than Normal"],
    ]
    story.append(styled_table(rows, [1.35 * inch, 1.25 * inch, 3.8 * inch]))
    story.append(Spacer(1, 7))
    story.extend(
        figure(
            "problem1_distribution.png",
            6.35 * inch,
            "Figure 1. Histogram, moment-matched Normal density, and the fitted Normal 1% quantile (dashed).",
        )
    )
    story.append(subsection("Fit", "Moment-matched Normal and its 1% quantile"))
    story.append(
        body(
            f"Matching the sample mean and variance gives N({r['mean']:.6f}, "
            f"{r['variance']:.8f}), with standard deviation {r['standard_deviation']:.6f}. "
            f"Its 1% quantile is {r['normal_q01']:.6f}. There are "
            f"<b>{r['observed_below_q01']}</b> observations below that threshold; a correctly "
            f"specified 1% tail model should produce about <b>{r['expected_below_q01']:.0f}</b> "
            f"out of {r['n']} observations."
        )
    )
    story.append(subsection("Reconcile", "What the Normal gets wrong"))
    story.append(
        body(
            "The fit confirms the prediction. The Normal underestimates the probability of a "
            "large negative observation: the empirical count is 2.6 times the model expectation. "
            "Matching variance cannot repair the shape mismatch. The sample has both a heavier "
            "left tail and left asymmetry, while the Normal must allocate equal tail probability "
            "to both sides. Used as a risk model, it would understate downside tail risk."
        )
    )


def add_problem_2(story: list) -> None:
    r = RESULTS["problem2"]
    story.append(PageBreak())
    story.append(Paragraph("2. A Regression Whose Errors Are Not Normal", styles["Section"]))
    story.append(subsection("Predict", "Scatter and OLS assumption"))
    story.append(
        body(
            "The pre-fit scatter shows a strong positive, approximately linear center with no "
            "obvious curvature. Most points stay close to the line, but several have unusually "
            "large vertical deviations. I therefore expected a symmetric heavy-tailed error, "
            "specifically Student-t, rather than a Normal error."
        )
    )
    story.extend(
        figure(
            "problem2_scatter.png",
            6.25 * inch,
            "Figure 2. The required pre-fit scatter. The few large vertical deviations motivate a heavy-tailed error model.",
        )
    )
    story.append(
        body(
            "This violates OLS assumption 7, normality of the error term. If assumptions 1-6 "
            "continue to hold, non-normality does not systematically bias the OLS slope: it "
            "remains unbiased/consistent. I therefore predicted little movement in beta. However, "
            "heavy tails make the slope less efficient and make exact finite-sample Normal/t "
            "inference invalid. Squared outliers can also make the usual OLS standard error noisy "
            "or inflated relative to a correctly specified heavy-tail likelihood."
        )
    )
    story.append(subsection("Fit", "OLS, Normal MLE, and Student-t MLE"))
    rows = [
        ["Model", "alpha", "beta", "Error parameters", "AICc"],
        [
            "OLS",
            fmt(r["ols"]["alpha"]),
            fmt(r["ols"]["beta"]),
            f"s={r['ols']['residual_standard_error']:.4f}",
            "N/A",
        ],
        [
            "Normal MLE",
            fmt(r["normal_mle"]["alpha"]),
            fmt(r["normal_mle"]["beta"]),
            f"sigma={r['normal_mle']['sigma']:.4f}",
            fmt(r["normal_mle"]["aicc"], 3),
        ],
        [
            "Student-t MLE",
            fmt(r["student_t_mle"]["alpha"]),
            fmt(r["student_t_mle"]["beta"]),
            f"scale={r['student_t_mle']['scale']:.4f}; df={r['student_t_mle']['degrees_freedom']:.3f}",
            fmt(r["student_t_mle"]["aicc"], 3),
        ],
    ]
    story.append(styled_table(rows, [1.1 * inch, 0.72 * inch, 0.72 * inch, 2.25 * inch, 0.8 * inch]))
    story.append(Spacer(1, 5))
    story.append(
        body(
            f"The OLS standard errors are SE(alpha)={r['ols']['se_alpha']:.4f} and "
            f"SE(beta)={r['ols']['se_beta']:.4f}. OLS itself does not specify a likelihood, so "
            "AICc is not defined for bare OLS. Under Normal errors, the MLE line equals the OLS "
            "line; its scale differs because the MLE divides residual sum of squares by n. "
            f"The Student-t AICc is lower by {r['aicc_difference_normal_minus_t']:.3f}, decisive "
            "evidence in favor of the Student-t error."
        )
    )
    story.extend(
        figure(
            "problem2_errors.png",
            6.25 * inch,
            "Figure 3. The Student-t fit is more peaked in the center and heavier in the far tails than the fitted Normal.",
        )
    )
    story.append(subsection("Reconcile", "Slope stability, model failure, and capital quantiles"))
    story.append(
        body(
            f"The beta estimates are {r['ols']['beta']:.4f} (OLS and Normal MLE) and "
            f"{r['student_t_mle']['beta']:.4f} (Student-t MLE), a difference of only "
            f"{abs(r['student_t_mle']['beta']-r['ols']['beta']):.4f}. This agrees with my "
            "prediction: the violation primarily changes the error model and efficiency, not the "
            "central slope. The rejected Normal is getting the residual shape wrong. Its single "
            "scale is pulled outward by extremes, making it too broad in the shoulders and too "
            "flat at the center, yet it still assigns too little probability to very remote errors."
        )
    )
    q95 = r["quantiles"]["0.95"]
    q995 = r["quantiles"]["0.995"]
    rows = [
        ["Upper error quantile", "Normal", "Student-t", "Wider"],
        ["95%", fmt(q95["normal"], 3), fmt(q95["student_t"], 3), "Normal"],
        ["99.5%", fmt(q995["normal"], 3), fmt(q995["student_t"], 3), "Student-t"],
    ]
    story.append(styled_table(rows, [1.65 * inch, 1.15 * inch, 1.15 * inch, 1.2 * inch]))
    story.append(Spacer(1, 5))
    story.append(
        Paragraph(
            "At 95%, the Normal is wider because its fitted sigma is inflated to compromise "
            "between the dense center and outliers. At 99.5%, the Student-t's polynomial tail "
            "dominates. For a high-confidence capital buffer, I would use the Student-t 99.5% "
            "quantile: it recognizes the extreme losses that the Normal understates. Because both "
            "fitted errors are symmetric, the corresponding lower-tail magnitudes are the same.",
            styles["Callout"],
        )
    )


def correlation_rows(matrix: dict, title: str) -> list[list]:
    names = ["x1", "x2", "x3", "x4"]
    rows = [[title] + names]
    for row_name in names:
        rows.append([row_name] + [f"{matrix[col][row_name]:.3f}" for col in names])
    return rows


def add_problem_3(story: list) -> None:
    r = RESULTS["problem3"]
    story.append(PageBreak())
    story.append(Paragraph("3. Pearson Against Spearman", styles["Section"]))
    story.append(subsection("Predict", "Pairs expected to agree and disagree"))
    story.append(
        body(
            "Before calculating correlations, the pair plot suggested that x1 and x2 would "
            "produce the largest disagreement: their relationship is nearly deterministic and "
            "monotone, but strongly curved. I expected Pearson and Spearman to agree more closely "
            "for x1-x3 because that cloud is approximately linear, and for every pair containing "
            "x4 because those clouds show no relationship, linear or monotone. I expected some "
            "secondary disagreement for x2-x3 because the x2 transformation bends an otherwise "
            "monotone relationship."
        )
    )
    story.extend(
        figure(
            "problem3_pairs.png",
            6.4 * inch,
            "Figure 4. Every pair, with Pearson and Spearman coefficients displayed above the diagonal.",
        )
    )
    story.append(subsection("Fit", "Correlation matrices"))
    story.append(styled_table(correlation_rows(r["pearson"], "Pearson"), [1.15 * inch] * 5))
    story.append(Spacer(1, 7))
    story.append(styled_table(correlation_rows(r["spearman"], "Spearman"), [1.15 * inch] * 5))
    story.append(Spacer(1, 6))
    story.append(
        body(
            f"The largest absolute gap is for x1-x2: Pearson={r['pearson']['x2']['x1']:.3f}, "
            f"Spearman={r['spearman']['x2']['x1']:.3f}, and the gap is {r['largest_gap']:.3f}."
        )
    )
    story.append(subsection("Reconcile", "Why the coefficients disagree"))
    story.append(
        body(
            "The plot shows x2 is approximately x1 cubed. Cubing preserves order, so nearly every "
            "pair of observations keeps the same rank and Spearman is close to one. The derivative "
            "of a cubic changes greatly across the range, so no single straight line represents "
            "the relationship equally well; Pearson is therefore lower. For the question 'how "
            "strong is the monotone association?', Spearman is the honest summary. Pearson is not "
            "wrong: it answers the narrower question 'how strong is the standardized linear "
            "covariation?' The observed results agree with the pre-fit prediction."
        )
    )


def add_problem_4(story: list) -> None:
    r = RESULTS["problem4"]
    cov = r["covariance"]
    story.append(PageBreak())
    story.append(Paragraph("4. Conditional Distributions", styles["Section"]))
    story.append(subsection("Predict", "Conditional variance and uncertainty reduction"))
    story.append(
        body(
            "I take the observed block to be X1=x1 and the target block to be X2=x2. Thus "
            "Sigma11=Var(x1), Sigma22=Var(x2), and Sigma12=Sigma21=Cov(x1,x2). The sample "
            "covariance matrix is:"
        )
    )
    rows = [
        ["", "x1", "x2"],
        ["x1", f"{cov['x1']['x1']:.6f}", f"{cov['x2']['x1']:.6f}"],
        ["x2", f"{cov['x1']['x2']:.6f}", f"{cov['x2']['x2']:.6f}"],
    ]
    story.append(styled_table(rows, [0.9 * inch, 1.35 * inch, 1.35 * inch]))
    story.append(Spacer(1, 6))
    story.append(
        body(
            "The multivariate-Normal partitioned result is<br/>"
            "<b>Var(X2 | X1=a) = Sigma22 - Sigma21 Sigma11<super>-1</super> Sigma12.</b><br/>"
            f"For this sample it is {cov['x2']['x2']:.6f} - "
            f"({cov['x1']['x2']:.6f})<super>2</super>/{cov['x1']['x1']:.6f} = "
            f"<b>{r['conditional_variance']:.6f}</b>. The remaining variance factor is "
            f"{r['conditional_variance']:.6f}/{cov['x2']['x2']:.6f} = "
            f"<b>{r['variance_ratio']:.4f}</b>, so conditioning reduces variance by "
            f"{100*r['variance_reduction_fraction']:.2f}%. In standard-deviation units, "
            f"uncertainty falls to sqrt({r['variance_ratio']:.4f}) = "
            f"{r['standard_deviation_ratio']:.4f} of its marginal level."
        )
    )
    story.append(
        body(
            "Under the multivariate Normal, this factor does not depend on the observed value a. "
            "The conditional variance formula contains only covariance blocks; a appears in the "
            "conditional mean, not in the conditional variance. I made this prediction before "
            "checking coverage against the data."
        )
    )
    story.append(subsection("Fit", "Conditional mean, band, and coverage"))
    story.append(
        body(
            "The corresponding conditional mean is<br/>"
            "<b>E[X2 | X1=a] = mu2 + Sigma21 Sigma11<super>-1</super>(a-mu1).</b><br/>"
            f"Numerically, E[x2|x1=a] = {r['means']['x2']:.6f} + "
            f"{r['conditional_slope']:.6f}(a - ({r['means']['x1']:.6f})). The coefficient "
            f"{r['conditional_slope']:.6f} is exactly Cov(x2,x1)/Var(x1), the simple OLS slope "
            "from regressing x2 on x1. The Normal 95% band uses conditional standard deviation "
            f"{r['conditional_standard_deviation']:.6f} and constant half-width "
            f"1.96s={r['band_half_width']:.6f}."
        )
    )
    story.extend(
        figure(
            "problem4_conditional.png",
            6.5 * inch,
            "Figure 5. Conditional expectation, constant-width 95% band, and empirical coverage by distance bucket.",
        )
    )
    rows = [
        ["Bucket", "n", "Inside band", "Residual SD"],
        ["All observations", str(r["n"]), f"{100*r['overall_coverage']:.2f}%", "-"],
        [
            "|x1-mean| < 1 SD",
            str(r["bucket_coverage"]["within_1_sd"]["n"]),
            f"{100*r['bucket_coverage']['within_1_sd']['coverage']:.2f}%",
            fmt(r["bucket_coverage"]["within_1_sd"]["residual_standard_deviation"], 3),
        ],
        [
            "1 <= |x1-mean|/SD < 2",
            str(r["bucket_coverage"]["between_1_and_2_sd"]["n"]),
            f"{100*r['bucket_coverage']['between_1_and_2_sd']['coverage']:.2f}%",
            fmt(r["bucket_coverage"]["between_1_and_2_sd"]["residual_standard_deviation"], 3),
        ],
        [
            "|x1-mean| >= 2 SD",
            str(r["bucket_coverage"]["beyond_2_sd"]["n"]),
            f"{100*r['bucket_coverage']['beyond_2_sd']['coverage']:.2f}%",
            fmt(r["bucket_coverage"]["beyond_2_sd"]["residual_standard_deviation"], 3),
        ],
    ]
    story.append(styled_table(rows, [2.4 * inch, 0.65 * inch, 1.15 * inch, 1.0 * inch]))
    story.append(subsection("Reconcile", "The failed constant-variance assumption"))
    story.append(
        body(
            f"Overall, {r['overall_inside_count']} of {r['n']} observations, or "
            f"{100*r['overall_coverage']:.2f}%, fall in the nominal 95% band. The aggregate is "
            "fairly close to 95%, but it hides a systematic pattern: coverage is 95.65% near the "
            "mean and only 85.79% and 90.20% in the two outer buckets. Residual standard deviation "
            "also rises from 1.076 near the center to about 1.59 away from it. This is evidence of "
            "heteroskedasticity: Var(x2|x1) changes with x1, violating the constant conditional "
            "variance implied by a multivariate Normal."
        )
    )
    story.append(
        body(
            "What survives is the fitted central line as the OLS best linear projection, and it "
            "remains the conditional mean if E[error|x1]=0 and the mean is truly linear. The slope "
            "interpretation as Cov(x2,x1)/Var(x1) also survives. What does not survive is the claim "
            "that covariance alone determines an exact Normal conditional distribution: the "
            "constant conditional variance, constant-width 95% band, and its uniform 95% coverage "
            "are invalid. Outside joint normality, the covariance formula is not automatically the "
            "exact conditional expectation; it is guaranteed only as the best linear predictor."
        )
    )


def add_problem_5(story: list) -> None:
    r = RESULTS["problem5"]
    story.append(PageBreak())
    story.append(Paragraph("5. Identifying an AR or MA Order", styles["Section"]))
    story.append(subsection("Predict", "ACF/PACF signature"))
    story.append(
        body(
            f"The approximate 95% significance band is +/-1.96/sqrt({r['n']}) = "
            f"+/-{r['significance_band']:.4f}. The ACF has a damped, oscillating pattern rather "
            "than a sharp cutoff. The PACF has two dominant spikes, lag 1 at "
            f"{r['pacf'][1]:.3f} and lag 2 at {r['pacf'][2]:.3f}, and then mostly cuts off. "
            "There are isolated later sample spikes, but not a persistent pattern; with many lags, "
            "some false 5% exceedances are expected. I therefore predicted an <b>AR(2)</b>. The "
            "rule used is: AR(p) has a decaying ACF and PACF cutoff after p; MA(q) reverses those "
            "roles."
        )
    )
    story.extend(
        figure(
            "problem5_acf_pacf.png",
            6.35 * inch,
            "Figure 6. Series, ACF, and PACF. The shaded region is the +/-1.96/sqrt(n) significance band.",
        )
    )
    story.append(subsection("Fit", "AICc comparison"))
    rows = [["Model", "AICc", "Key coefficients"]]
    for name in ["AR(1)", "AR(2)", "AR(3)", "MA(1)", "MA(2)", "MA(3)"]:
        model = r["models"][name]
        coefficient_text = ", ".join(
            f"{key}={value:.4f}"
            for key, value in model["parameters"].items()
            if key != "sigma2" and key != "const"
        )
        rows.append([name, f"{model['aicc']:.3f}", coefficient_text])
    story.append(styled_table(rows, [1.0 * inch, 1.05 * inch, 3.8 * inch]))
    story.append(Spacer(1, 6))
    story.append(
        body(
            f"The smallest AICc is {r['models']['AR(2)']['aicc']:.3f} for AR(2), so the fitted "
            "model selection agrees with the pre-fit ACF/PACF prediction. Its coefficients are "
            f"phi1={r['models']['AR(2)']['parameters']['ar.L1']:.4f} and "
            f"phi2={r['models']['AR(2)']['parameters']['ar.L2']:.4f}."
        )
    )
    story.append(subsection("Reconcile", "AR(2) versus AR(3) and the role of complexity"))
    story.append(
        body(
            f"AR(3) estimates phi3={r['models']['AR(3)']['parameters']['ar.L3']:.4f}, which is "
            "very small. Adding it improves log likelihood only from "
            f"{r['models']['AR(2)']['log_likelihood']:.3f} to "
            f"{r['models']['AR(3)']['log_likelihood']:.3f}, a gain of just "
            f"{r['models']['AR(3)']['log_likelihood']-r['models']['AR(2)']['log_likelihood']:.3f}. "
            "AICc rewards better likelihood but charges for every fitted parameter, with an extra "
            "finite-sample correction. The tiny likelihood gain cannot pay for the third AR "
            f"coefficient, so AICc rises from {r['models']['AR(2)']['aicc']:.3f} to "
            f"{r['models']['AR(3)']['aicc']:.3f}. Plain R-squared has no parameter penalty and "
            "cannot decrease when an additional regressor is added. It therefore would not reject "
            "AR(3) on parsimony grounds, even though the added coefficient contributes almost no "
            "explanatory value."
        )
    )


def add_methods(story: list) -> None:
    story.append(PageBreak())
    story.append(Paragraph("Methods and Reproducibility", styles["Section"]))
    story.append(
        body(
            "All calculations were produced by analysis.py from the five supplied CSV files. "
            "The script uses pandas/numpy for data handling, scipy for distributions and maximum "
            "likelihood, statsmodels for OLS and ARIMA estimation, and matplotlib for figures. "
            "AICc counts every fitted distribution/model parameter and uses AICc = AIC + "
            "2k(k+1)/(n-k-1), matching the Week 2 convention."
        )
    )
    rows = [
        ["Quantity", "Convention"],
        ["Variance", "Unbiased sample variance (ddof=1), except MLE scales"],
        ["Skewness", "Bias-corrected standardized sample skewness"],
        ["Kurtosis", "Bias-corrected excess kurtosis (Normal = 0)"],
        ["Student-t", "Location-scale t; df constrained above 2"],
        ["PACF", "Yule-Walker, no sample-size adjustment (statsmodels ywm)"],
        ["AR/MA fits", "Gaussian ARIMA likelihood with a constant and innovation variance"],
    ]
    story.append(styled_table(rows, [1.45 * inch, 4.75 * inch]))
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "The README in this submission gives the exact commands required to install "
            "dependencies, reproduce results.json and every figure, and rebuild this PDF.",
            styles["Callout"],
        )
    )


def build() -> None:
    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.68 * inch,
        bottomMargin=0.72 * inch,
        title="Assignment 1 - Univariate and Multivariate Statistics",
        author="Samuel Ma",
        subject="FinTech 545 - Quantitative Risk Management",
    )

    story = [
        Spacer(1, 0.95 * inch),
        Paragraph("ASSIGNMENT 1", styles["Subtitle"]),
        Paragraph("Univariate and Multivariate Statistics", styles["ReportTitle"]),
        Spacer(1, 0.12 * inch),
        Table(
            [
                [small("COURSE"), body("FinTech 545 - Quantitative Risk Management")],
                [small("STUDENT"), body("Samuel Ma")],
                [small("DATE"), body("September 4, 2026")],
            ],
            colWidths=[1.05 * inch, 4.65 * inch],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.35, colors.HexColor("#DCE4EC")),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            ),
        ),
        Spacer(1, 0.5 * inch),
        Paragraph("Approach", styles["Section"]),
        body(
            "Each problem is organized as Predict, Fit, and Reconcile. Predictions are based "
            "only on the exploratory view requested in the prompt; numerical model results are "
            "then reported separately and compared back to those predictions."
        ),
        Paragraph(
            "Headline findings: the Normal understates the left tail in Problem 1; Student-t "
            "errors dominate Normal errors in Problem 2; rank correlation correctly captures a "
            "curved monotone pair in Problem 3; the constant-width conditional band fails away "
            "from the center in Problem 4; and both ACF/PACF and AICc select AR(2) in Problem 5.",
            styles["Callout"],
        ),
    ]

    add_problem_1(story)
    add_problem_2(story)
    add_problem_3(story)
    add_problem_4(story)
    add_problem_5(story)
    add_methods(story)

    document.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    print(PDF_PATH)


if __name__ == "__main__":
    build()
