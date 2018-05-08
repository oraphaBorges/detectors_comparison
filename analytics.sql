create table stats(
    kase text,
    name text,
    pair number,
    iteration number,
    status text,
    kps_feat_min number,
    kps_feat_max number,
    kps_feat_diff_mean number,
    kps_feat_diff_std number,
    matches_origin number,
    matches number,
    amount_below_error number
);

SELECT
    kase,
    name,
    pair,
    iteration,
    status,
    kps_feat_min,
    kps_feat_max,
    kps_feat_diff_mean,
    kps_feat_diff_std,
    matches_origin,
    matches,
    amount_below_error
FROM
    datas_171219_115013
GROUP BY
    technique, situation, phase
ORDER BY
    technique, situation, phase;
