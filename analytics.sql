SELECT technique, situation,avg(kp1),avg(kp2),avg(matches),avg(time),avg(anglesMean),avg(anglesSD),avg(scaleMean),avg(scaleSD) FROM datas_171219_115013 GROUP BY technique, situation ORDER BY technique, situation;

