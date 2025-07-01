# Code for: Assessing the Real-World Performance of Xylazine Test Strips for Community-Based Drug Checking in Los Angeles

## Setup
rm(list = ls())
pacman::p_load(data.table, tidyverse, ggplot2,ggrepel, grid, gridExtra,lubridate,cowplot,stringr,forecast,Hmisc,zoo,table1,caret,haven)
root <- "SET WD"
aesth <- theme_bw() + theme(axis.title = element_text(size=10,face='bold'),axis.text =element_text(size=10,face='bold'),plot.title =element_text(size=12,face='bold'),strip.background = element_rect(fill="white"),strip.text=element_text(size=10,face='bold'),legend.position = 'top')

mk_ORs <- function(mod){
  coeff <- data.table(var=row.names(summary(mod)$coefficients[-1,]),summary(mod)$coefficients[-1,])
  names(coeff) <- c("variable","coeff","se","z","p")
  coeff[,OR:=exp(coeff)]
  coeff[,OR_lwr:=exp(coeff-(1.96*se))]
  coeff[,OR_upr:=exp(coeff+(1.96*se))]
  coeff[,signifigant:=ifelse(p<.05,"1","0")]  
  return(coeff)  
}

#################################################################################################
dt <- data.table(read_dta(paste0(root,"REDACTED.dta")))

dt[,xylazine_percent:=as.numeric(xylazine_percent)]

dt[,date1:=str_split_i(recorded_date,pattern = " ",i = 1)]
dt[,date1:=str_replace(date1,"2025","25")]
dt[,date1:=str_replace(date1,"2024","24")]
dt[,date1:=str_replace(date1,"2023","23")]

dt[,date:=as.Date(date1,"%m/%d/%y")]

nrow(dt[date>"2023-06-01"])

dt[,yq:=as.yearqtr(date,format="%Y-%m-%d")]
dt[,yq:=format(dt$yq,format="%Y Q%q")]
dt[,pid:=.I]

#remove non-wisebatch
dt[date<"2023-06-08",xylstrip:=NA]
dt[xylstrip=="negative",xystrip:=0]
dt[xylstrip=="positive",xystrip:=1]

confusionMatrix(factor(dt$xylazine_dumb),factor(dt$xystrip),positive="1")

ft <- dt[fentanyl_dumb==1]
ft.xy.mo <- ft[,.(xy=mean(xylazine_dumb),n=uniqueN(pid)),by=.(yq)]
ft.xy.mo[,qid:=.I]

ft[,qid:=factor(yq,
levels=c("2023 Q1","2023 Q2","2023 Q3","2023 Q4","2024 Q1","2024 Q2","2024 Q3","2024 Q4","2025 Q1"),
labels=seq(1:9))]
ft[,qid:=as.numeric(qid)]

table(ft$xylstrip)
ft[xylstrip=="negative",xystrip:=0]
ft[xylstrip=="positive",xystrip:=1]

#2x2 table. sensitivity/specificity PPV, NPV
confusionMatrix(factor(ft$xylazine_dumb),factor(ft$xystrip),positive="1")

#histogram of positivity by concentration
gg_s_1 <- ggplot(dt[xylstrip%in%c("positive","negative")],aes(x=xylazine_percent,fill=xylstrip)) +
geom_histogram()+aesth +
labs(x="Percent Xylazine Concentration",y="Count",
title="Xylazine Test Strip Result by Xylazine Concentration")+
scale_fill_brewer("Test Strip Result",palette = 6,type='qual',direction=1,labels=c("False Negative","True Positive"))+
scale_x_continuous(breaks=seq(0,30,2.5),labels=paste0(seq(0,30,2.5),"%"))

pdf(paste0(root,"REDACTED",Sys.Date(),".pdf"),width=8,height=4)
gg_s_1
dev.off()  


