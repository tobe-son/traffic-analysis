# 文献調査

## 調査方針

学会・査読会議で公表された論文を中心に、交通音データセット、音響による速度推定、未知収録条件への一般化を調査した。

| 文献 | 研究上の知見 | 本研究での扱い |
| --- | --- | --- |
| Abeßer et al., EUSIPCO 2021, IDMT-Traffic | 2種類のマイクで収録した4,718通過事象を公開し、車種・進行方向分類を比較 | 機器差をドメインとみなし、未知条件評価が必要である根拠 |
| Djukanović et al., TELFOR 2022, VS13 | 13車両・400通過の速度付き音声映像と車両単位CVを提案 | leave-one-vehicle-out を採用し、同一車両の漏洩を防ぐ |
| Heittola et al., DCASE 2020 | 複数デバイスと低計算量条件で音響シーン分類の一般化を評価 | 平均性能だけでなく未知機器・未知環境を分離して評価 |
| Kim et al., DCASE 2021, Residual Normalization | 周波数方向 instance normalization と shortcut によりデバイス固有情報を抑制 | 本年度の正規化 ablation の根拠 |
| Cho et al., DCASE 2019 Workshop | 異環境の同一クラスを近づける metric learning で未知環境性能を改善 | 昨年度の距離学習を環境一般化の観点から再検証 |

## 先行研究の問題点と研究ギャップ

公開ベンチマークは増えたが、シミュレーションで学習した速度回帰器が実環境へ移る際の性能低下を、車両・地点の独立性を保った分割と信頼区間で検証した例は限られる。昨年度のランダム分割では、同じ地点や車両に由来する音が両集合に入る可能性を排除できず、実運用性能を過大評価する恐れがある。また、潜在空間で地点を分類しにくいことだけでは、速度情報を保った環境不変表現かどうかを証明できない。

そこで本研究では、未知地点・未知車両を単位とする group-wise 検証を主評価とし、速度性能と環境不変性を同時に測る。正規化と雑音拡張は別々に ablation し、改善要因を切り分ける。

## 参考文献

1. J. Abeßer et al., “IDMT-Traffic: An Open Benchmark Dataset for Acoustic Traffic Monitoring Research,” EUSIPCO, 2021. https://arxiv.org/abs/2104.13620
2. S. Djukanović, N. Bulatović, and I. Čavor, “A Dataset for Audio-Video Based Vehicle Speed Estimation,” TELFOR, 2022. https://arxiv.org/abs/2212.01651
3. T. Heittola, A. Mesaros, and T. Virtanen, “Acoustic Scene Classification in DCASE 2020 Challenge: Generalization Across Devices and Low Complexity Solutions,” DCASE, 2020. https://arxiv.org/abs/2005.14623
4. B. Kim et al., “Domain Generalization on Efficient Acoustic Scene Classification Using Residual Normalization,” DCASE, 2021. https://arxiv.org/abs/2111.06531
5. J. Cho et al., “Acoustic Scene Classification Based on a Large-margin Factorized CNN,” DCASE Workshop, 2019. https://arxiv.org/abs/1910.06784

