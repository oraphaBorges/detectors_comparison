SELECT
    technique,
    situation,
    phase,
    avg(kps1),
    avg(kps2),
    avg(matches),
    avg(time),
    avg(anglesDiffMean),
    avg(anglesDiffStd),
    avg(kpsRatioMean),
    avg(kpsRatioStd)
FROM
    datas_171219_115013
GROUP BY
    technique, situation, phase
ORDER BY
    technique, situation, phase;
