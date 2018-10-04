clear
cd "C:\Users\maria\Desktop\MÁSTER\Segundo año\PRIMER CUATRIMESTRE\Quantitative Macroeconomics\Problem sets\ps2"
use data_gross.dta

**************************Extention with gross labor share for U.S.A******************************

* Unambigous capital income (UCI)
gen UCI = Rincom + Cprof + Ninteres + CsurplusGov 

* Unambigous income (UI)
gen UI = UCI + DEP + CE

* Proportion of capital that we will use to estimate the capital part of ambiguos income(thita).
gen theta = (UCI + DEP)/UI

* Ambigous income (AI), Computed without statistical discrepancy.
gen AI = PI + T - S + Bctrans

* Ambigous capital income (ACI)
gen ACI = AI*theta

* Capital income (CI)
gen CI = ACI + DEP + UCI

* Output (Y)
gen Y = UCI + DEP + CE + AI

* Gross labor share (LSgross)
gen LSgross = 1-(CI/Y)

* Plot LSgross:
line LSgross Year, saving(grossLS) 


