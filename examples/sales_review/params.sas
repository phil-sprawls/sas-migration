/*------------------------------------------------------------------
  Sales review - run parameters.  Business users edit this file only.
------------------------------------------------------------------*/
%let report_title = Furniture sales review;                       /* title on the report */
%let start_month  = 01APR1993;                                    /* first month included */
%let end_month    = 31MAR1994;                                    /* last month included */
%let countries    = 'CANADA','GERMANY','U.S.A.';                  /* countries to include */
%let row_filter   = actual > 100 and prodtype ne 'OFFICE';        /* sales rows to include */
%let flag_filter  = pct_of_target < 95;                           /* groups to flag for review */
%let keep_cols    = country region product month actual predict;  /* columns of the detail table */
%let group_cols   = country, product;                             /* summary grouping */
