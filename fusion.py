def fuse_predictions(disease_confidence, suitability_score):
    """
    Combines CNN and RF outputs into a final health status.
    """

    final_score = (0.6 * disease_confidence) + (0.4 * suitability_score)

    if final_score >= 0.75:
        status = "Healthy"
        recommendation = (
            "Crop conditions are stable. Maintain current practices."
        )

    elif final_score >= 0.5:
        status = "At Risk"
        recommendation = (
            "Monitor crop closely. Consider improving nutrition or disease control."
        )

    else:
        status = "Critical"
        recommendation = (
            "Immediate intervention required. Improve soil conditions and treat disease."
        )

    return {
        "final_score": round(final_score, 2),
        "status": status,
        "recommendation": recommendation
    }
