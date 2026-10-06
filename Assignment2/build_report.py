"""Build the final Assignment 2 PDF from computed results and figures."""

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
FIGURES = ROOT / "figures"
PDF_PATH = ROOT / "assignment2answer.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2878A3")
TEAL = colors.HexColor("#269A91")
ORANGE = colors.HexColor("#D96C45")
LIGHT_BLUE = colors.HexColor("#EAF3F8")
LIGHT_TEAL = colors.HexColor("#EAF7F5")
TEXT = colors.HexColor("#243447")
MUTED = colors.HexColor("#64748B")


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24, leading=29, textColor=NAVY, alignment=TA_LEFT, spaceAfter=12))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontSize=12.5, leading=16, textColor=BLUE, spaceAfter=6))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16.5, leading=20, textColor=NAVY, spaceBefore=4, spaceAfter=8, keepWithNext=True))
    styles.add(ParagraphStyle(name="Subsection", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.2, leading=14, textColor=BLUE, spaceBefore=7, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=9.1, leading=13.0, textColor=TEXT, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=7.5, leading=9.8, textColor=TEXT))
    styles.add(ParagraphStyle(name="SmallHeader", parent=styles["Small"], fontName="Helvetica-Bold", textColor=colors.white))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=7.6, leading=9.5, textColor=MUTED, alignment=TA_CENTER, spaceAfter=7))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontSize=9.0, leading=12.7, textColor=TEXT, leftIndent=9, rightIndent=9, borderColor=TEAL, borderWidth=1, borderPadding=7, backColor=LIGHT_TEAL, spaceBefore=4, spaceAfter=8))
    return styles


STYLES = make_styles()


def body(text: str) -> Paragraph:
    return Paragraph(text, STYLES["Body"])


def subsection(label: str, title: str) -> Paragraph:
    return Paragraph(f"{label}. {title}", STYLES["Subsection"])


def table(rows, widths):
    converted = []
    for row_index, row in enumerate(rows):
        style = STYLES["SmallHeader"] if row_index == 0 else STYLES["Small"]
        converted.append([Paragraph(str(cell), style) for cell in row])
    result = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C7D6")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7FAFC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return result


def figure(filename: str, width: float, caption: str):
    image = Image(str(FIGURES / filename))
    image.drawHeight = image.imageHeight * width / image.imageWidth
    image.drawWidth = width
    return [image, Paragraph(caption, STYLES["Caption"])]


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def money(value):
    number = float(value)
    return f"-${abs(number):,.2f}" if number < 0 else f"${number:,.2f}"


def percent(value, digits=2):
    return f"{100.0 * float(value):.{digits}f}%"


def callout(text: str) -> Paragraph:
    return Paragraph(text, STYLES["Callout"])


def matrix_rows(payload: dict, digits: int = 3) -> list[list[str]]:
    labels = payload["columns"]
    rows = [[""] + labels]
    for label, values in zip(payload["rows"], payload["values"]):
        rows.append([label] + [fmt(value, digits) for value in values])
    return rows


def add_problem_1(story: list, r: dict) -> None:
    story.append(PageBreak())
    story.append(Paragraph("1. Correlations from Mismatched Histories", STYLES["Section"]))
    story.append(subsection("Predict", "Observation overlap and PSD risk"))
    story.append(body(
        "Pairwise estimation can produce a non-PSD matrix because every cell may use a "
        "different sample. Complete-case estimation standardizes one common data matrix, so "
        "its correlation matrix is a Gram matrix and is PSD apart from roundoff. Because IDX "
        "is almost a linear combination of A, B, C, and D, I predicted a near-zero smallest "
        "eigenvalue and unusual sensitivity to even small pairwise inconsistencies."
    ))
    counts = r["joint_observation_counts"]
    story.append(table(matrix_rows(counts, 0), [0.65 * inch] + [0.82 * inch] * 5))
    story.append(Spacer(1, 5))
    story.append(body(
        f"Only <b>{r['sample']['complete_all_five_rows']}</b> days contain all five series. "
        "Before calculating correlations, I expected pairs containing D to be least reliable: "
        "C-D has 83 joint days, A-D and B-D have 93, and D-IDX has 96."
    ))
    story.append(subsection("Fit", "Complete-case and pairwise matrices"))
    story.append(Paragraph("Complete-case correlation (81 rows)", STYLES["Subsection"]))
    story.append(table(matrix_rows(r["complete_case"]["correlation"]), [0.65 * inch] + [0.82 * inch] * 5))
    story.append(Spacer(1, 5))
    story.append(body(
        "Complete-case eigenvalues: "
        + ", ".join(fmt(x, 6) for x in r["complete_case"]["eigenvalues_ascending"])
        + ". Ordinary Cholesky succeeds. The minimum eigenvalue is positive but small, "
        "confirming the near-singularity prediction."
    ))
    story.append(Paragraph("Pairwise correlation", STYLES["Subsection"]))
    story.append(table(matrix_rows(r["pairwise"]["correlation"]), [0.65 * inch] + [0.82 * inch] * 5))
    story.append(Spacer(1, 5))
    story.append(body(
        "Pairwise eigenvalues: "
        + ", ".join(fmt(x, 6) for x in r["pairwise"]["eigenvalues_ascending"])
        + f". The minimum is <b>{r['pairwise']['minimum_eigenvalue']:.8f}</b>, so ordinary "
        "Cholesky fails. Using each series' full-history sample standard deviation and "
        "Sigma = diag(sd) R diag(sd), the tracking weights (-0.4,-0.3,-0.2,-0.1,1) "
        f"produce variance <b>{r['pairwise']['tracking_portfolio_variance']:.10f}</b>."
    ))
    full_sd = r["pairwise"]["full_history_standard_deviations"]
    story.append(body(
        "Full-history sample SDs used in the reconstruction are "
        + ", ".join(f"{name}={value:.6f}" for name, value in full_sd.items())
        + "."
    ))
    story.append(callout(
        "PSD means w' Sigma w >= 0 for every weight vector w. The specified tracking vector gives "
        "w' Sigma w < 0, so Sigma is not PSD; equivalently, its negative eigenvalue guarantees a "
        "direction with a negative quadratic form. This is a matrix inconsistency, not an economic variance."
    ))
    story.append(subsection("Fit", "Correlation repairs"))
    repairs = r["repairs"]
    repair_rows = [["Method", "Min eigenvalue", "Frobenius distance", "Max cell move", "Tracking variance"]]
    for key, label in (("rebonato_jackel", "Rebonato-Jackel"), ("higham", "Higham")):
        item = repairs[key]
        minimum_display = (
            "0 (roundoff -6.24e-15)" if key == "higham"
            else f"{item['minimum_eigenvalue']:.2e}"
        )
        repair_rows.append([
            label,
            minimum_display,
            fmt(item["frobenius_distance_from_pairwise"], 8),
            fmt(item["maximum_absolute_off_diagonal_change"], 8),
            f"{item['tracking_portfolio_variance']:.8e}",
        ])
    story.append(table(repair_rows, [1.25 * inch, 1.05 * inch, 1.22 * inch, 1.05 * inch, 1.25 * inch]))
    story.append(Spacer(1, 6))
    story.append(body(
        "Rebonato-Jackel clips negative eigenvalues and rescales the spectral root to a unit "
        "diagonal. Higham uses alternating PSD and unit-diagonal projections with Dykstra "
        f"correction; it converged in {repairs['higham']['iterations']} iterations. Both repairs "
        "restore a small positive tracking variance and are very close to one another. Higham's "
        "reported -6.24e-15 minimum eigenvalue is floating-point zero, not a material PSD failure."
    ))
    story.append(subsection("Reconcile", "What moved, and what mattered most"))
    moves = repairs["higham"]["largest_off_diagonal_changes"][:5]
    move_rows = [["Entry", "Pairwise", "Higham", "Signed move"]]
    for item in moves:
        move_rows.append([
            f"{item['row']}-{item['column']}",
            fmt(item["original"], 6),
            fmt(item["repaired"], 6),
            fmt(item["signed_change"], 6),
        ])
    story.append(table(move_rows, [1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch]))
    story.append(Spacer(1, 6))
    story.append(body(
        "The largest Higham move is A-IDX, even though that pair has 242 observations; C-D "
        "has only 83 but moves less. Higham was not given observation counts. It uses global "
        "eigenstructure and equal Frobenius cost per cell, changing the collection of entries "
        "that most cheaply removes the negative direction."
    ))
    gap = r["estimator_gap"]
    story.append(callout(
        f"The complete-case/pairwise distance is {gap['complete_case_vs_pairwise_frobenius_distance']:.6f}, "
        f"or {gap['gap_to_higham_repair_distance_ratio']:.2f} times the Higham repair distance. "
        "For these data, estimator choice matters much more than choosing between the two repairs."
    ))


def add_problem_2(story: list, r: dict) -> None:
    story.append(PageBreak())
    story.append(Paragraph("2. A Volatility Estimate After the Regime Changed", STYLES["Section"]))
    story.append(subsection("Predict", "The recent regime should dominate EW risk"))
    story.append(body(
        "I computed arithmetic returns, removed the full-sample mean, and inspected the plot "
        "before calculating VaR. Volatility rises sharply near day 461, so I classified the "
        "last 40 returns as the recent regime. I predicted, from smallest to largest: equal-weight "
        "Normal, historical, Student-t, EW(0.97), EW(0.94). Full-sample skew and heavy tails should "
        "put historical and Student-t beyond a Normal benchmark, with the exact order of those two "
        "least certain. The EW measures should be largest because recent volatility is higher, and "
        "lambda=0.94 reacts most quickly."
    ))
    story.extend(figure("problem2_demeaned_returns.png", 6.45 * inch, "Figure 1. Demeaned arithmetic returns. The shaded final 40 days define the visually selected high-volatility regime."))
    moments = r["moments"]
    moment_rows = [
        ["Mean", "Variance", "SD", "Skewness", "Excess kurtosis"],
        [f"{moments['mean']:.2e}", fmt(moments["variance"], 8), fmt(moments["standard_deviation"], 6), fmt(moments["skewness"], 3), fmt(moments["excess_kurtosis"], 3)],
    ]
    story.append(table(moment_rows, [0.95 * inch, 1.05 * inch, 0.95 * inch, 0.95 * inch, 1.15 * inch]))
    story.append(Spacer(1, 6))
    story.append(body(
        "Excess kurtosis of 2.166 is incompatible with a single stationary Normal regime as a "
        "credible description. The prediction is about stationarity, not merely whether a Normal "
        "sample could ever show a nonzero sample moment."
    ))
    story.append(PageBreak())
    story.append(subsection("Predict (continued)", "Effective memory of the two decay factors"))
    ew_rows = [["lambda", "Effective n", "Half-life", "Weight on last 40", "EW SD", "VaR"]]
    for key in ("lambda_0.94", "lambda_0.97"):
        item = r["exponentially_weighted"][key]
        ew_rows.append([
            fmt(item["decay"], 2), fmt(item["finite_sample_effective_n"], 2),
            f"{item['half_life_days']:.2f} days", percent(item["weight_on_most_recent_40_days"], 1),
            fmt(item["standard_deviation"], 5), money(item["var_usd"]),
        ])
    story.append(table(ew_rows, [0.65 * inch, 0.85 * inch, 1.0 * inch, 1.15 * inch, 0.85 * inch, 1.05 * inch]))
    story.append(Spacer(1, 6))
    story.append(body(
        "The plot suggests 40 recent days are different. Lambda=0.94 assigns 91.6% of its total "
        "weight to them, compared with 70.4% for lambda=0.97, reinforcing the prediction that 0.94 "
        "will produce the larger risk estimate."
    ))
    story.append(subsection("Fit", "Fitted t and five VaRs"))
    var_order = r["fit_ranking_smallest_to_largest"]
    labels = {
        "student_t_mle": "Student-t MLE", "normal_equal_weight": "Equal-weight Normal",
        "historical_full_sample": "Historical", "normal_ew_lambda_0.97": "EW Normal 0.97",
        "normal_ew_lambda_0.94": "EW Normal 0.94",
    }
    var_rows = [["Rank", "Method", "5% one-day VaR (loss)"]]
    for index, key in enumerate(var_order, 1):
        var_rows.append([str(index), labels[key], money(r["var_usd"][key])])
    story.append(table(var_rows, [0.55 * inch, 2.4 * inch, 1.25 * inch]))
    tfit = r["student_t_fit"]
    story.append(Spacer(1, 6))
    story.append(body(
        f"The Student-t MLE is location={tfit['location']:.8f}, scale={tfit['scale']:.8f}, "
        f"and nu={tfit['degrees_of_freedom']:.4f}. Scale is not SD; the implied SD is "
        f"{tfit['implied_standard_deviation_if_df_gt_2']:.8f}. Historical VaR uses the "
        "course RiskStats order-statistic convention, here the 25th smallest return."
    ))
    story.append(subsection("Reconcile", "A regime mixture, not one persistent heavy tail"))
    regimes = r["regime"]["moments_by_regime"]
    regime_rows = [["Regime", "Days", "SD", "Skewness", "Excess kurtosis"]]
    for key, label in (("earlier", "Earlier"), ("recent", "Recent")):
        item = regimes[key]
        regime_rows.append([label, str(item["n_days"]), fmt(item["standard_deviation"], 6), fmt(item["skewness"], 3), fmt(item["excess_kurtosis"], 3)])
    story.append(table(regime_rows, [1.1 * inch, 0.7 * inch, 1.0 * inch, 1.0 * inch, 1.15 * inch]))
    story.append(Spacer(1, 6))
    story.append(body(
        f"Recent SD is {r['regime']['standard_deviation_ratio_recent_to_earlier']:.3f} times earlier SD, "
        "while each regime's excess kurtosis is near zero. Mixing their variances implies excess "
        f"kurtosis about {r['regime']['zero_mean_normal_mixture_implied_excess_kurtosis']:.3f}. "
        "The fitted t is mainly a one-distribution approximation to a volatility-regime mixture."
    ))
    noise = r["ew_noise_comparison"]
    story.append(body(
        f"The EW VaR gap is {money(noise['absolute_var_gap_usd'])}; the VaR-equivalent SEs are "
        f"{money(noise['lambda_0.94_var_standard_error_usd'])} and "
        f"{money(noise['lambda_0.97_var_standard_error_usd'])}. Thus the gap is "
        f"{noise['gap_over_lambda_0.94_standard_error']:.2f} and "
        f"{noise['gap_over_lambda_0.97_standard_error']:.2f} times those individual SEs, and only "
        f"{noise['gap_over_root_sum_square_noise_scale']:.2f} times the conservative root-sum-square "
        "noise scale. It is not convincingly larger than estimation noise. <b>Opinion:</b> I would "
        "use 0.94 for a responsive trading limit and 0.97 for a more stable capital number."
    ))
    story.append(callout(
        "Reconciliation: EW(0.94) > EW(0.97) agrees with the prediction, but the actual full-sample "
        "order is Student-t < Normal < historical, not Normal < historical < Student-t. A fitted t "
        "is both more peaked in the center and heavier in the far tails. At 5% its smaller fitted "
        "scale dominates; much farther out, its polynomial tail would dominate."
    ))


def add_problem_3(story: list, r: dict) -> None:
    story.append(PageBreak())
    story.append(Paragraph("3. When Diversification Raises VaR", STYLES["Section"]))
    story.append(subsection("Predict", "A probability jump crosses the 5% cutoff"))
    story.append(body(
        "The return distributions have a large nondefault profit cluster and a separate default "
        "loss cluster. Both bonds are extremely left-skewed and leptokurtic, so Normal VaR will "
        "smooth a discrete default jump into an ordinary standard deviation and cannot describe "
        "the actual tail mechanism."
    ))
    moment_rows = [["Bond", "Mean", "Variance", "Skewness", "Excess kurtosis"]]
    for bond in ("A", "B"):
        item = r["moments"][bond]
        moment_rows.append([bond, fmt(item["mean"], 6), fmt(item["variance"], 6), fmt(item["skewness"], 3), fmt(item["excess_kurtosis"], 3)])
    story.append(table(moment_rows, [0.7 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.25 * inch]))
    counts = r["large_loss_counts"]
    story.append(Spacer(1, 6))
    story.append(body(
        f"A loses more than 20% in {counts['A']} scenarios ({percent(counts['A_rate'])}); B in "
        f"{counts['B']} ({percent(counts['B_rate'])}); at least one does in {counts['at_least_one']} "
        f"({percent(counts['at_least_one_rate'])}); both do in only {counts['both']} "
        f"({percent(counts['both_rate'])}). Each standalone default rate is below 5%, but the union "
        "rate exceeds 5%, so diversification should raise 5% VaR. Only 17 cases are joint defaults: "
        "the diversified worst 5% is therefore dominated by one $1m bond defaulting, while the "
        "concentrated tail exposes the full $2m to A's default. I predicted lower diversified ES."
    ))
    story.extend(figure("problem3_pnl_distributions.png", 6.45 * inch, "Figure 2. Concentration has fewer but more severe default states; diversification creates a more frequent one-default loss region."))
    story.append(PageBreak())
    story.append(subsection("Fit", "Historical and Normal risk"))
    position_labels = {
        "one_million_A": "$1m A", "one_million_B": "$1m B",
        "two_million_A": "$2m A", "one_million_each": "$1m A + $1m B",
    }
    rows = [["Position", "Hist VaR 5%", "Hist ES 5%", "Normal VaR", "Hist VaR 1%", "Hist ES 1%"]]
    for key in ("one_million_A", "one_million_B", "two_million_A", "one_million_each"):
        item = r["positions"][key]
        rows.append([
            position_labels[key], money(item["historical"]["5pct"]["var"]), money(item["historical"]["5pct"]["es"]),
            money(item["normal_var_5pct"]), money(item["historical"]["1pct"]["var"]), money(item["historical"]["1pct"]["es"]),
        ])
    story.append(table(rows, [1.02 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch]))
    story.append(Spacer(1, 6))
    story.append(body(
        "Negative 5% VaR for a standalone bond means its 5th-percentile scenario is still a "
        "profit; it is not clipped to zero. ES remains large because it averages the rarer "
        "default states beyond that quantile."
    ))
    story.append(subsection("Reconcile", "VaR fails subadditivity; ES does not"))
    sub = r["subadditivity_5pct"]
    story.append(body(
        f"At 5%, combined VaR is {money(sub['var']['combined_A_plus_B'])}, while the two standalone "
        f"VaRs sum to {money(sub['var']['standalone_A_plus_standalone_B'])}. VaR subadditivity "
        f"fails by {money(sub['var']['lhs_minus_rhs'])}. Combined ES is "
        f"{money(sub['es']['combined_A_plus_B'])}, below the standalone sum "
        f"{money(sub['es']['standalone_A_plus_standalone_B'])}; ES has "
        f"{money(-sub['es']['lhs_minus_rhs'])} of subadditivity slack."
    ))
    story.append(body(
        "Both portfolio-level predictions are confirmed: diversification raises VaR from "
        f"{money(r['positions']['two_million_A']['historical']['5pct']['var'])} to "
        f"{money(r['positions']['one_million_each']['historical']['5pct']['var'])}, but lowers ES "
        f"from {money(r['positions']['two_million_A']['historical']['5pct']['es'])} to "
        f"{money(r['positions']['one_million_each']['historical']['5pct']['es'])}."
    ))
    compare = r["choice_comparison"]
    story.append(body(
        f"Normal VaR favors diversification ({money(compare['normal_5pct']['diversified_var'])} "
        f"versus {money(compare['normal_5pct']['concentrated_var'])}), but for the wrong reason: "
        "it rewards lower variance while missing the probability jump. At 1%, historical VaR "
        f"reverses: diversified {money(compare['historical_1pct']['diversified_var'])} versus "
        f"concentrated {money(compare['historical_1pct']['concentrated_var'])}. The 1% cutoff lies "
        "inside each bond's default region, so both portfolios now reveal default risk and splitting "
        "notional reduces severity. VaR's failure is confidence-level dependent."
    ))


def add_problem_4(story: list, r: dict) -> None:
    story.append(PageBreak())
    story.append(Paragraph("4. Gaussian or Student-t Copula", STYLES["Section"]))
    story.append(subsection("Predict", "Margins and empirical tail clustering"))
    rows = [["Series", "Mean", "Variance", "Skewness", "Excess kurtosis", "Prediction"]]
    predictions = {"X1": "Normal", "X2": "Student-t", "X3": "Student-t"}
    for name in ("X1", "X2", "X3"):
        item = r["moments"][name]
        rows.append([name, fmt(item["mean"], 6), fmt(item["variance"], 7), fmt(item["skewness"], 3), fmt(item["excess_kurtosis"], 3), predictions[name]])
    story.append(table(rows, [0.65 * inch, 0.9 * inch, 1.0 * inch, 0.85 * inch, 1.05 * inch, 1.05 * inch]))
    sensitivity = r["single_observation_sensitivity"]
    story.append(Spacer(1, 6))
    story.append(body(
        f"X2's kurtosis of 7.125 looks exceptional. Observation {sensitivity['X2']['removed_observation_number_1_based']} "
        f"contains X2={percent(sensitivity['X2']['removed_value'], 3)} and is also extreme for the other margins. "
        f"Removing it lowers X2 kurtosis to {sensitivity['X2']['moments_without_observation']['excess_kurtosis']:.3f} "
        f"and X3 kurtosis to {sensitivity['X3']['moments_without_observation']['excess_kurtosis']:.3f}; both remain "
        "positive, so the Student-t predictions do not depend on one point alone."
    ))
    pairs = r["rank_analysis"]["pairs"]
    tail_rows = [["Pair", "Observed lower", "Observed upper", "Independence"]]
    for key in ("X1-X2", "X1-X3", "X2-X3"):
        item = pairs[key]
        tail_rows.append([key, str(item["observed_lower_count"]), str(item["observed_upper_count"]), fmt(item["independence_expected_each_tail"], 3)])
    story.append(table(tail_rows, [1.0 * inch, 1.2 * inch, 1.2 * inch, 1.1 * inch]))
    story.append(Spacer(1, 5))
    story.append(body(
        "Independence expects only 0.625 days per pair and side; observed counts range from 6 to "
        "12. The rank plots show a broadly related but diffuse middle, while the corner clusters "
        "look unusually dense. Together with the common outlier, that suggested dependence beyond "
        "ordinary middle correlation, so I predicted that the Student-t copula would win."
    ))
    story.extend(figure("problem4_rank_pairs.png", 6.65 * inch, "Figure 3. Pairwise pseudo-observations rank/(n+1); highlighted corners show joint 2.5% tail days."))
    story.append(PageBreak())
    story.append(subsection("Fit", "Marginal model selection"))
    margin_rows = [["Series", "Normal AICc", "t nu", "t AICc", "Selected", "AICc advantage"]]
    for name in ("X1", "X2", "X3"):
        item = r["margin_fits"][name]
        t_nu = "Normal limit" if item["student_t"]["degrees_of_freedom"] > 1e6 else fmt(item["student_t"]["degrees_of_freedom"], 3)
        selected_label = "Normal" if item["selected_by_aicc"] == "normal" else "Student-t"
        margin_rows.append([
            name, fmt(item["normal"]["aicc"], 2), t_nu,
            fmt(item["student_t"]["aicc"], 2), selected_label, fmt(item["aicc_advantage_of_selected"], 2),
        ])
    story.append(table(margin_rows, [0.65 * inch, 1.0 * inch, 0.85 * inch, 1.0 * inch, 1.0 * inch, 1.15 * inch]))
    story.append(Spacer(1, 6))
    story.append(body(
        "X1's fitted t has effectively infinite degrees of freedom, so it is the Normal limit and "
        "cannot justify an extra parameter. AICc selects Normal for X1 and Student-t for X2 and X3."
    ))
    selected_rows = [["Series", "Selected fit", "Location", "Scale", "nu"]]
    for name in ("X1", "X2", "X3"):
        item = r["margin_fits"][name]
        selected = item["selected_by_aicc"]
        fit = item["normal"] if selected == "normal" else item["student_t"]
        selected_rows.append([
            name,
            "Normal" if selected == "normal" else "Student-t",
            fmt(fit["location"], 7),
            fmt(fit["scale"], 7),
            "n/a" if selected == "normal" else fmt(fit["degrees_of_freedom"], 3),
        ])
    story.append(table(selected_rows, [0.7 * inch, 1.15 * inch, 1.15 * inch, 1.15 * inch, 0.8 * inch]))
    story.append(Spacer(1, 6))
    copula = r["copula_fit"]
    dependence_rows = [["Pair", "Kendall tau", "rho = sin(pi*tau/2)"]]
    for pair, first, second in (("X1-X2", 0, 1), ("X1-X3", 0, 2), ("X2-X3", 1, 2)):
        dependence_rows.append([
            pair,
            fmt(copula["kendall_tau_matrix"][first][second], 5),
            fmt(copula["correlation_matrix_used"][first][second], 5),
        ])
    story.append(body(
        "For copula fitting, I transformed each raw return with its selected fitted marginal CDF. "
        "These parametric uniforms differ from the rank/(n+1) values used only for the exploratory plot and counts."
    ))
    story.append(table(dependence_rows, [1.0 * inch, 1.25 * inch, 1.75 * inch]))
    story.append(Spacer(1, 6))
    copula_rows = [["Copula", "Log likelihood", "Parameters", "AICc", "BIC"]]
    for key, label in (("gaussian", "Gaussian"), ("student_t", "Student-t")):
        item = copula[key]
        copula_rows.append([label, fmt(item["log_likelihood"], 3), str(item["n_parameters"]), fmt(item["aicc"], 3), fmt(item["bic"], 3)])
    story.append(table(copula_rows, [1.1 * inch, 1.2 * inch, 1.0 * inch, 1.1 * inch, 1.1 * inch]))
    story.append(Spacer(1, 6))
    story.append(body(
        f"The Kendall-tau correlation matrix is already positive definite. The Student-t copula "
        f"profiles to nu={copula['student_t']['degrees_of_freedom']:.4f}. It wins AICc by "
        f"{copula['student_t_aicc_advantage']:.3f} and BIC by {copula['student_t_bic_advantage']:.3f}, "
        "decisive evidence for tail dependence."
    ))
    risk = r["portfolio_risk_usd"]
    risk_rows = [["Model", "VaR 5%", "ES 5%", "VaR 1%", "ES 1%"]]
    for key, label in (("historical", "Historical"), ("gaussian_copula", "Gaussian copula"), ("student_t_copula", "Student-t copula")):
        item = risk[key]
        risk_rows.append([label, money(item["var_5pct"]), money(item["es_5pct"]), money(item["var_1pct"]), money(item["es_1pct"])])
    story.append(table(risk_rows, [1.35 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch]))
    story.append(PageBreak())
    story.append(subsection("Reconcile", "What the t copula fixes"))
    tail_compare = [["Pair", "Obs L", "G L", "t L", "Obs U", "G U", "t U"]]
    for key in ("X1-X2", "X1-X3", "X2-X3"):
        item = pairs[key]
        tail_compare.append([
            key, item["observed_lower_count"], fmt(item["gaussian_copula_lower_days_per_1000"], 2), fmt(item["student_t_copula_lower_days_per_1000"], 2),
            item["observed_upper_count"], fmt(item["gaussian_copula_upper_days_per_1000"], 2), fmt(item["student_t_copula_upper_days_per_1000"], 2),
        ])
    story.append(table(tail_compare, [0.85 * inch] + [0.76 * inch] * 6))
    story.append(Spacer(1, 6))
    gauss = risk["gaussian_copula"]
    tcop = risk["student_t_copula"]
    observed_total = sum(
        item["observed_lower_count"] + item["observed_upper_count"] for item in pairs.values()
    )
    gaussian_total = sum(
        item["gaussian_copula_lower_days_per_1000"] + item["gaussian_copula_upper_days_per_1000"]
        for item in pairs.values()
    )
    student_total = sum(
        item["student_t_copula_lower_days_per_1000"] + item["student_t_copula_upper_days_per_1000"]
        for item in pairs.values()
    )
    story.append(body(
        f"Across the six pair-tail cells, the data contain {observed_total} joint days, versus "
        f"{gaussian_total:.2f} implied by the Gaussian and {student_total:.2f} by the Student-t. "
        "Thus the t copula is much closer to the observed aggregate. Its 5% VaR falls by only "
        f"{money(abs(tcop['var_5pct'] - gauss['var_5pct']))}; the 5% cutoff is not deep enough for "
        "simultaneous extremes to dominate. The largest movement is 1% ES, which rises by "
        f"{money(tcop['es_1pct'] - gauss['es_1pct'])}, because ES averages losses beyond an "
        "already extreme cutoff."
    ))
    ll = r["likelihood_contributions"]
    story.append(body(
        f"Daily log-density differences sum to {ll['total_t_minus_gaussian']:.4f}. The "
        f"{ll['central_day_count']} days on which every empirical rank lies between 5% and 95% "
        f"contribute {ll['central_day_contribution']:.4f}, or {percent(ll['central_share_of_total'])} "
        "of the total. Information criteria score every observation, not only hand-picked tail days."
    ))
    td = r["tail_dependence"]
    story.append(callout(
        f"For the most correlated pair, {td['most_correlated_pair']} (rho={td['rho']:.3f}), the "
        f"fitted Student-t lower and upper tail-dependence coefficient is {td['student_t_lower_and_upper']:.4f}. "
        "The Gaussian value is exactly zero for correlation below one."
    ))


def add_problem_5(story: list, r: dict) -> None:
    story.append(PageBreak())
    story.append(Paragraph("5. Model-Based Simulation and Residual Correlation", STYLES["Section"]))
    story.append(subsection("Predict", "The covariance cross-term changes sign"))
    story.append(body(
        "If A and B share an omitted industry exposure, Cov(e_A,e_B) is positive. In "
        "Var(w_A r_A + w_B r_B), the cross-term is 2 w_A w_B Cov(r_A,r_B). P1 has same-sign "
        "weights, so setting residual covariance to zero must understate its VaR. P2 has "
        "opposite-sign weights, so the same shortcut removes a hedge and must overstate VaR."
    ))
    story.append(subsection("Fit", "OLS and two residual simulations"))
    story.append(body(
        f"The {r['n_prices']} price observations produce {r['n_returns']} arithmetic returns for each series."
    ))
    ols_rows = [["Stock", "alpha", "beta", "Residual SD"]]
    for name in ("A", "B"):
        item = r["ols"][name]
        ols_rows.append([name, fmt(item["alpha"], 8), fmt(item["beta"], 6), fmt(item["residual_standard_deviation"], 8)])
    story.append(table(ols_rows, [0.8 * inch, 1.2 * inch, 1.1 * inch, 1.25 * inch]))
    story.append(Spacer(1, 6))
    story.append(body(
        f"Residual correlation is <b>{r['residual_correlation']:.6f}</b>, large and positive. "
        "The assignment assumes zero expected returns, so I report the fitted intercepts as OLS "
        "diagnostics but omit alpha from the risk simulation; market and residual draws are centered. "
        "Full and diagonal-residual simulations use the same 100,000 Gaussian draws."
    ))
    rows = [["Portfolio", "Full residual VaR", "Independent VaR", "Independent - full", "Delta-normal"]]
    for key, label in (("P1_long_A_long_B", "P1: long A + long B"), ("P2_long_A_short_B", "P2: long A - short B")):
        item = r["portfolios"][key]
        sim = item["simulation_var_5pct"]
        analytic = item["analytic_normal_var_5pct"]
        rows.append([label, money(sim["full_residual_covariance"]), money(sim["diagonal_residual_covariance"]), money(sim["diagonal_minus_full"]), money(analytic["direct_sample_stock_covariance_delta_normal"])])
    story.append(table(rows, [1.55 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.1 * inch]))
    story.extend(figure("problem5_residual_correlation.png", 6.65 * inch, "Figure 4. Ignoring positive residual correlation narrows P1 but widens the long-short P2 distribution."))
    story.append(subsection("Reconcile", "The shortcut removes risk from P1 and a hedge from P2"))
    p1 = r["portfolios"]["P1_long_A_long_B"]["simulation_var_5pct"]
    p2 = r["portfolios"]["P2_long_A_short_B"]["simulation_var_5pct"]
    p1_analytic = r["portfolios"]["P1_long_A_long_B"]["analytic_normal_var_5pct"]
    p2_analytic = r["portfolios"]["P2_long_A_short_B"]["analytic_normal_var_5pct"]
    story.append(body(
        f"The direction matches the prediction. Independence lowers P1 VaR by "
        f"{money(abs(p1['diagonal_minus_full']))} ({abs(p1['diagonal_minus_full_pct_of_full']):.2f}%), "
        f"but raises P2 VaR by {money(p2['diagonal_minus_full'])} "
        f"({p2['diagonal_minus_full_pct_of_full']:.2f}%). P2 is more exposed because common "
        "residual movements hedge a long-short spread; setting covariance to zero removes that hedge."
    ))
    story.append(body(
        f"The factor reconstruction and direct stock covariance agree to "
        f"{r['maximum_absolute_covariance_reconstruction_error']:.2e}. Under OLS, residuals are "
        "sample-orthogonal to the market, so Cov(r)=beta beta' Var(MKT)+Sigma_e. With joint "
        "Normal components and linear P&amp;L, factor simulation and delta-normal describe the same "
        f"distribution. The full simulation is {money(abs(p1_analytic['simulation_full_minus_delta_normal']))} "
        f"below delta-normal for P1 and {money(abs(p2_analytic['simulation_full_minus_delta_normal']))} "
        "below it for P2; these small differences are Monte Carlo quantile noise. They would "
        "separate with nonlinear positions, heavy tails, heteroskedasticity, time-varying betas, "
        "or residual dependence on the market."
    ))


def add_methods(story: list, results: dict) -> None:
    story.append(PageBreak())
    story.append(Paragraph("Methods and Reproducibility", STYLES["Section"]))
    story.append(body(
        "All calculations are generated from the five supplied CSV files by run_assignment.py. "
        "The reusable risk545 package contains the model logic; build_report.py reads the generated "
        "JSON and figures, so the written numbers and code output share one source."
    ))
    rows = [
        ["Quantity", "Convention"],
        ["Returns", "Arithmetic P[t]/P[t-1]-1"],
        ["Moments", "Sample variance (n-1); corrected skewness and excess kurtosis"],
        ["Historical VaR", "Course RiskStats floor/ceil one-based order-statistic average"],
        ["Historical ES", "Negative mean P&L at or below the VaR cutoff, including ties"],
        ["AICc/BIC", "Every estimated margin or copula parameter counted"],
        ["Exploratory ranks", "Average rank/(n+1), strictly inside (0,1), for plots and tail counts"],
        ["Copula uniforms", "Selected fitted marginal CDF applied to each raw return"],
        ["Simulation", f"100,000 draws; master seed {results['metadata']['seed']}"],
    ]
    story.append(table(rows, [1.45 * inch, 4.85 * inch]))
    story.append(Spacer(1, 10))
    story.append(callout(
        "Reproduction: create a Python environment, install requirements.txt, run "
        "python run_assignment.py --seed 545, then run python -m pytest -q. The README contains "
        "the complete commands."
    ))


def page_decoration(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8E2EA"))
    canvas.line(0.7 * inch, 0.54 * inch, 7.8 * inch, 0.54 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.72 * inch, 0.34 * inch, "FinTech 545 - Assignment 2")
    canvas.drawRightString(7.78 * inch, 0.34 * inch, f"Page {document.page}")
    canvas.restoreState()


def build() -> None:
    results = json.loads((OUTPUT / "results.json").read_text())
    story = build_story(results)
    document = SimpleDocTemplate(
        str(PDF_PATH), pagesize=letter, rightMargin=0.72 * inch,
        leftMargin=0.72 * inch, topMargin=0.67 * inch, bottomMargin=0.72 * inch,
        title="Assignment 2 - Covariance, VaR, and Copulas", author="Samuel Ma",
    )
    document.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    print(f"Wrote {PDF_PATH}")


def build_story(results: dict) -> list:
    """Assemble the complete Predict/Fit/Reconcile report."""

    story = [
        Spacer(1, 1.0 * inch),
        Paragraph("ASSIGNMENT 2", STYLES["Subtitle"]),
        Paragraph("Covariance, VaR, and Copulas", STYLES["ReportTitle"]),
        Spacer(1, 0.10 * inch),
        table([
            ["COURSE", "FinTech 545 - Quantitative Risk Management"],
            ["STUDENT", "Samuel Ma"],
            ["DATE", "October 5, 2026"],
        ], [1.0 * inch, 4.7 * inch]),
        Spacer(1, 0.35 * inch),
        Paragraph("Approach", STYLES["Section"]),
        body(
            "Each problem is organized as Predict, Fit, and Reconcile. Predictions are based only "
            "on the requested exploratory evidence; fitted results then test those predictions. "
            "Risk numbers use positive dollar losses and the course finite-sample VaR convention."
        ),
        callout(
            "Headline results: mismatched pairwise histories create a negative portfolio variance; "
            "a volatility regime shift dominates EW VaR; diversification raises 5% VaR but lowers "
            "ES for defaultable bonds; a Student-t copula decisively captures joint tails; and "
            "positive residual correlation raises long-long risk while reducing long-short risk."
        ),
    ]
    add_problem_1(story, results["problem1"])
    add_problem_2(story, results["problem2"])
    add_problem_3(story, results["problem3"])
    add_problem_4(story, results["problem4"])
    add_problem_5(story, results["problem5"])
    add_methods(story, results)
    return story


if __name__ == "__main__":
    build()
