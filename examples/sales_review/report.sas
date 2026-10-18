/*------------------------------------------------------------------
  Sales review (synthetic example for sas2dbx)
  Data:       SALES.PRODUCTSALES - a copy of SASHELP.PRDSALE (public, productsales.sas7bdat)
  Parameters: params.sas, edited by business users
------------------------------------------------------------------*/
%include "params.sas";

/* sales targets per country and product; some products have no target */
data targets;
  infile datalines dsd missover;
  length country $10 product $10;
  input country $ product $ target;
datalines;
CANADA,SOFA,30000
CANADA,BED,26000
GERMANY,SOFA,25000
GERMANY,CHAIR,24000
U.S.A.,SOFA,31000
U.S.A.,BED,
;
run;

/* the sales rows in scope */
data detail;
  set sales.productsales;
  where &row_filter and month between "&start_month"d and "&end_month"d;
  if country in (&countries);
  keep &keep_cols;
run;

proc sort data=detail;
  by country product month;
run;

/* totals per group */
proc sql;
  create table summary as
  select &group_cols, count(*) as n_rows, sum(actual) as actual, sum(predict) as predict
  from detail
  group by &group_cols;
quit;

proc sort data=summary;
  by country product;
run;

proc sort data=targets;
  by country product;
run;

/* against target: no target gives a missing percentage */
data review;
  merge summary(in=in_sales) targets;
  by country product;
  if in_sales;
  pct_of_target = round(100 * actual / target, 0.1);
run;

/* groups to review */
data flagged;
  set review;
  where &flag_filter;
run;

data report_title;
  length title $60;
  title = "&report_title";
run;
