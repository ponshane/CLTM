# Project Structure
```
-- src/
    -- codebase/
    -- jupyter/
    -- out/
        -- CLTM_Inputs/
        -- PLDA_Inputs/
        -- Mallet_Mono_LDA/
        -- Experiments_result/
```

* src directory 所有本專案的程式碼、實驗方式(py, bash, commands...)、輸出結果
    * codebase 則為所有模組化的程式碼（反覆使用易跑實驗）
    * jupyter 包含本專案的 jupyter 檔
    * out 包含所有本專案的程式輸出以及實驗結果
        * CLTM_Inputs, PLDA_Inputs 皆為第三方套件的輸入需求檔
        * Mallet_Mono_LDA 準備 PM_LDA 的中間產出檔
        * Experiments_result 紀錄本專案的所有衡量結果
