#Code for: High Variation in Purity of Consumer-Level Illicit Fentanyl Samples in Los Angeles, 2023-2025
## Setup
rm(list = ls())
pacman::p_load(data.table, tidyverse, ggplot2,ggrepel, grid, gridExtra,lubridate,cowplot,stringr,forecast,Hmisc,zoo,table1,caret,ggExtra,haven,labelled,interpretCI)
root <- "SET_WD"
aesth <- theme_bw() + theme(axis.title = element_text(size=10,face='bold'),axis.text =element_text(size=10,face='bold'),plot.title =element_text(size=12,face='bold'),strip.background = element_rect(fill="white"),strip.text=element_text(size=10,face='bold'),legend.position = 'top')

#################################################################################################
#load data

#select drugs to include in table and figure
c.drugs <- fread(paste0(root,"REDACTED.csv"))
c.drugs[,variable:=Substance]

dt <- data.table(read_dta(paste0(root,"data/REDACTED.dta")))

dt[,fent_fluoro_sum:=NULL]
dt[!is.na(fentanyl_percent)&!is.na(fluorofentanyl_percent),fent_fluoro_sum:=fentanyl_percent+fluorofentanyl_percent]
dt[!is.na(fentanyl_percent)&is.na(fluorofentanyl_percent),fent_fluoro_sum:=fentanyl_percent]
dt[is.na(fentanyl_percent)&!is.na(fluorofentanyl_percent),fent_fluoro_sum:=fluorofentanyl_percent]

dt[!is.na(fent_fluoro_sum)&fent_fluoro_sum<1.0,ffcat:="<1%"]
dt[!is.na(fent_fluoro_sum)&fent_fluoro_sum>1.0&fent_fluoro_sum<5.0,ffcat:="1-5%"]
dt[!is.na(fent_fluoro_sum)&fent_fluoro_sum>5.0&fent_fluoro_sum<15.0,ffcat:="5-15%"]
dt[!is.na(fent_fluoro_sum)&fent_fluoro_sum>15,ffcat:=">15%"]

dt[,date1:=str_split_i(recorded_date,pattern = " ",i = 1)]
dt[,date1:=str_replace(date1,"2025","25")]
dt[,date1:=str_replace(date1,"2024","24")]
dt[,date1:=str_replace(date1,"2023","23")]

dt[,date:=as.Date(date1,"%m/%d/%y")]
dt[,yq:=as.yearqtr(date,format="%Y-%m-%d")]
dt[,yq:=format(dt$yq,format="%Y Q%q")]
dt[,pid:=.I]

dt[,qid:=factor(yq,
                levels=c("2023 Q1","2023 Q2","2023 Q3","2023 Q4","2024 Q1","2024 Q2","2024 Q3","2024 Q4","2025 Q1","2025 Q2"),
                labels=seq(1:10))]
dt[,qid:=as.numeric(qid)]




#table of concentration by drug expectation, pill vs powder, presence vs absence of key drugs of interest 

    
#expected drug
dt[drugexpectation%in%c("benzodiazepines","cocaine","fentanyl","heroin"),ex:=str_to_title(drugexpectation)]  
dt[drugexpectation%in%c("oxycodone","percocet"),ex:="Prescription Opioids"]  
dt[drugexpectation%in%c("fentanyl, xylazine"),ex:="Xylazine+Fentanyl"]  
dt[!is.na(fent_fluoro_sum)&is.na(ex),ex:="Other/Declined"]

tb.ex <- dt[!is.na(ex)&!is.na(fent_fluoro_sum),c("fent_fluoro_sum","ex")]
setnames(tb.ex,"ex","value")
tb.ex[,variable:="Expected Substances"]

#form
table(dt[!is.na(fent_fluoro_sum),form2])
dt[,form2:=to_character(form2)]
#dt[form2=="",form2:="Other/Missing"]
tb.fo <- dt[!is.na(form2)&!is.na(fent_fluoro_sum),c("fent_fluoro_sum","form2")]
setnames(tb.fo,"form2","value")
tb.fo[,variable:="Sample Form"]


#presence of key other drugs
tb.su <- dt[!is.na(fent_fluoro_sum)]
tb.su <- tb.su[,c("fent_fluoro_sum",names(dt)[names(dt)%like%"_dumb"]),with=F]
tb.su <- melt.data.table(tb.su,id.vars = "fent_fluoro_sum")
tb.su[,variable:=str_trim(str_to_title(str_replace_all(variable,"_dumb|_"," ")))]
tb.su[,value:=ifelse(value==1,"Present","Absent")]
tb.su <- merge(tb.su,c.drugs,by='variable')
tb.su2 <- tb.su[Class%in%c("Fentanyl Analog","Non-Fentanyl Drug","Bulking Agent")]
tb.su2 <- tb.su2[order(Class,variable)]

tb.all <- dt[!is.na(fent_fluoro_sum),c("fent_fluoro_sum")]
tb.all[,variable:="Overall"]
tb.all[,value:="Overall"]


tb <- rbind(tb.all,tb.ex,tb.fo,tb.su2,ignore.attr=TRUE,fill=T)
tb[,obs:=1]

tb1 <- tb[,.(n=sum(obs),mean=mean(fent_fluoro_sum,na.rm=T),median=median(fent_fluoro_sum,na.rm=T),sd=sd(fent_fluoro_sum,na.rm=T),min=min(fent_fluoro_sum,na.rm=T),max=max(fent_fluoro_sum,na.rm=T)),by=.(variable,value)]

tb1[,variable:=factor(variable,levels=unique(tb1$variable))]
tb1[,ord:=as.numeric(variable)]
tb1 <- tb1[order(ord,-n)]
tb1[,ord:=NULL]
tb1[,value:=paste0(value, " (n=",n,")")]
tb1[,`mean(sd; range)`:=paste0(sprintf(mean,fmt = '%#.1f'),"% (",sprintf(sd,fmt = '%#.1f'),"% ; ",sprintf(min,fmt = '%#.1f'),"% - ",sprintf(max,fmt = '%#.1f'),"%)")]

fwrite(tb1[,c("variable","value","mean(sd; range)")],paste0(root,"visuals/table1_concentration_",Sys.Date(),".csv"))

stb <- rbind(tb.all,tb.ex,tb.fo,tb.su,ignore.attr=TRUE,fill=T)
stb[,obs:=1]

stb1 <- stb[,.(n=sum(obs),mean=mean(fent_fluoro_sum,na.rm=T),median=median(fent_fluoro_sum,na.rm=T),sd=sd(fent_fluoro_sum,na.rm=T),min=min(fent_fluoro_sum,na.rm=T),max=max(fent_fluoro_sum,na.rm=T)),by=.(variable,value)]

stb1[,variable:=factor(variable,levels=unique(stb1$variable))]
stb1[,ord:=as.numeric(variable)]
stb1 <- stb1[order(ord,-n)]
stb1[,ord:=NULL]
stb1[,value:=paste0(value, " (n=",n,")")]
stb1[,`mean(sd; range)`:=paste0(sprintf(mean,fmt = '%#.1f'),"% (",sprintf(sd,fmt = '%#.1f'),"% ; ",sprintf(min,fmt = '%#.1f'),"% - ",sprintf(max,fmt = '%#.1f'),"%)")]

fwrite(stb1[,c("variable","value","mean(sd; range)")],paste0(root,"visuals/supp_table1_concentration_",Sys.Date(),".csv"))


#regressions
mod <- lm(fent_fluoro_sum~date,data=dt)
summary(mod)



#graphs
ft <- dt[!is.na(yq)&!is.na(fent_fluoro_sum)]
ft[yq<"2024 Q2",yq:="2024 Q1"]

ft.mo <- ft[,.(fentf_pc=mean(fent_fluoro_sum,na.rm=T),n=uniqueN(pid)),by=.(yq)]
ft.mo[,qid:=.I]

#fentanyl concentration over time
gg1 <- ggplot(ft.mo,aes(y=fentf_pc,x=yq)) + aesth+
  #geom_smooth(aes(y=fentf_pc,x=qid,group="1"),method='lm',se=T,color="#d8b365",alpha=.1,size=2)+
  geom_bar(stat='identity',fill="#1f78b4")+ 
  geom_label(aes(label=paste0("n=",n)),y=-2)+
  geom_text(aes(label=paste0(sprintf(fentf_pc,fmt = '%#.1f'),"%"),y=fentf_pc+2))+
  scale_y_continuous(breaks=seq(0,100,5),labels=paste0(seq(0,100,5),"%"))+
  scale_x_discrete(labels=c("2024 Q1\nand Prior","2024 Q2","2024 Q3","2024 Q4","2025 Q1","2025 Q2"))+
  labs(y="Concentration(%)",x="Quarter",title="A) Time Trend")+
  expand_limits(y = c(-3, 20))+
  geom_hline(yintercept = 0,size=1)


#distribution of fentanyl concentration
c.col <- "#1f78b4"
gg2 <- ggplot(dt,aes(x=fent_fluoro_sum))+aesth+
  geom_density(color=c.col,fill=c.col,size=3,alpha=.3)+
    labs(y="",x="\n% Concentration",title="B) Distribution") +scale_x_continuous(breaks=seq(0,100,10),labels=paste0(seq(0,100,10),"%"))+
  geom_vline(xintercept=mean(dt$fent_fluoro_sum,na.rm=T),alpha=.99,color=c.col)+
  geom_vline(xintercept=median(dt$fent_fluoro_sum,na.rm=T),alpha=.99,linetype='longdash',color=c.col)+
  theme(axis.text.y = element_blank())+
  scale_y_continuous(expand = c(0,0))


pdf(paste0(root,"visuals/figure_1_",Sys.Date(),".pdf"),width=12,height=3)
grid.arrange(gg1,gg2,nrow=1)
  dev.off()  


#scatter plots of relationships
conc <- dt[!is.na(fent_fluoro_sum),c("pid","fent_fluoro_sum","heroin_percent","xylazine_percent","btmps_percent","lidocaine_percent","cocaine_percent","x_4_anpp_percent","methamphetamine_percent","phenethyl_4_anpp_percent")]
conc.l <- melt.data.table(conc,id.vars = c("pid","fent_fluoro_sum"))
conc.l[,value:=as.numeric(value)]
conc.l <- conc.l[!is.na(value)]


#Heroin
c.col <- "#2b8cbe"
c.drug <- "Heroin"

gg <- ggplot(conc.l[variable=="heroin_percent"],aes(y=fent_fluoro_sum,x=value))+aesth+
   geom_smooth(method='lm',alpha=.3,fill=c.col,color=c.col)+
  #facet_wrap(~variable,scales='free')+
   geom_point(shape=21,stroke=1.1,,size=2,color="black",fill=c.col)+
   theme(legend.position = 'none')+
   scale_y_continuous(breaks=seq(0,60,2),labels=paste0(seq(0,60,2),"%"))+
   scale_x_continuous(breaks=seq(0,60,10),labels=paste0(seq(0,60,10),"%"))+
  labs(y="Fentanyl and Fluorofentanyl",x=c.drug,title=paste0("\nA) ",c.drug))

gg3a <- ggMarginal(gg, type = "density",fill=c.col,alpha=0.7)

#BTMPS
c.col <- "#33a02c"
c.drug <- "BTMPS"

gg <- ggplot(conc.l[variable=="btmps_percent"],aes(y=fent_fluoro_sum,x=value))+aesth+
  geom_smooth(method='lm',alpha=.3,fill=c.col,color=c.col)+
  #facet_wrap(~variable,scales='free')+
  geom_point(shape=21,stroke=1.1,,size=2,color="black",fill=c.col)+
  theme(legend.position = 'none')+
  scale_y_continuous(breaks=seq(0,60,5),labels=paste0(seq(0,60,5),"%"))+
  scale_x_continuous(breaks=seq(0,60,10),labels=paste0(seq(0,60,10),"%"))+
  labs(y="Fentanyl and Fluorofentanyl",x=c.drug,title=paste0("\nB) ",c.drug))

gg3b <- ggMarginal(gg, type = "density",fill=c.col,alpha=0.7)

#Xylazine
c.col <- "#6a3d9a"
c.drug <- "Xylazine"

gg <- ggplot(conc.l[variable=="xylazine_percent"],aes(y=fent_fluoro_sum,x=value))+aesth+
  geom_smooth(method='lm',alpha=.3,fill=c.col,color=c.col)+
  #facet_wrap(~variable,scales='free')+
  geom_point(shape=21,stroke=1.1,,size=2,color="black",fill=c.col)+
  theme(legend.position = 'none')+
  scale_y_continuous(breaks=seq(0,60,5),labels=paste0(seq(0,60,5),"%"))+
  scale_x_continuous(breaks=seq(0,60,10),labels=paste0(seq(0,60,10),"%"))+
  labs(y="Fentanyl and Fluorofentanyl",x=c.drug,title=paste0("\nC) ",c.drug))

gg3c <- ggMarginal(gg, type = "density",fill=c.col,alpha=0.7)

#Meth
c.col <- "#ff7f00"
c.drug <- "Methamphetamine"

gg <- ggplot(conc.l[variable=="methamphetamine_percent"],aes(y=fent_fluoro_sum,x=value))+aesth+
  geom_smooth(method='lm',alpha=.3,fill=c.col,color=c.col)+
  #facet_wrap(~variable,scales='free')+
  geom_point(shape=21,stroke=1.1,,size=2,color="black",fill=c.col)+
  theme(legend.position = 'none')+
  scale_y_continuous(breaks=seq(0,60,1),labels=paste0(seq(0,60,1),"%"))+
  scale_x_continuous(breaks=seq(0,60,2),labels=paste0(seq(0,60,2),"%"))+
  labs(y="Fentanyl and Fluorofentanyl",x=c.drug,title=paste0("D) ",c.drug))

gg3d <- ggMarginal(gg, type = "density",fill=c.col,alpha=0.7)


#Cocaine
c.col <- "#e31a1c"
c.drug <- "Cocaine"

gg <- ggplot(conc.l[variable=="cocaine_percent"],aes(y=fent_fluoro_sum,x=value))+aesth+
  geom_smooth(method='lm',alpha=.3,fill=c.col,color=c.col)+
  #facet_wrap(~variable,scales='free')+
  geom_point(shape=21,stroke=1.1,,size=2,color="black",fill=c.col)+
  theme(legend.position = 'none')+
  scale_y_continuous(breaks=seq(0,60,5),labels=paste0(seq(0,60,5),"%"))+
  scale_x_continuous(breaks=seq(0,60,2),labels=paste0(seq(0,60,2),"%"))+
  labs(y="Fentanyl and Fluorofentanyl",x=c.drug,title=paste0(".",c.drug))

gg3e <- ggMarginal(gg, type = "density",fill=c.col,alpha=0.7)
gg3e

#lidocaine
c.col <- "#fb9a99"
c.drug <- "Lidocaine"

gg <- ggplot(conc.l[variable=="lidocaine_percent"],aes(y=fent_fluoro_sum,x=value))+aesth+
  geom_smooth(method='lm',alpha=.3,fill=c.col,color=c.col)+
  #facet_wrap(~variable,scales='free')+
  geom_point(shape=21,stroke=1.1,,size=2,color="black",fill=c.col)+
  theme(legend.position = 'none')+
  scale_y_continuous(breaks=seq(0,60,5),labels=paste0(seq(0,60,5),"%"))+
  scale_x_continuous(breaks=seq(0,60,5),labels=paste0(seq(0,60,5),"%"))+
  labs(y="Fentanyl and Fluorofentanyl",x=c.drug,title=paste0("E) ",c.drug))

gg3f <- ggMarginal(gg, type = "density",fill=c.col,alpha=0.7)
gg3f

#"x_4_anpp_percent"
c.col <- "#a6cee3"
c.drug <- "4ANPP"

gg <- ggplot(conc.l[variable=="x_4_anpp_percent"],aes(y=fent_fluoro_sum,x=value))+aesth+
  geom_smooth(method='lm',alpha=.3,fill=c.col,color=c.col)+
  #facet_wrap(~variable,scales='free')+
  geom_point(shape=21,stroke=1.1,,size=2,color="black",fill=c.col)+
  theme(legend.position = 'none')+
  scale_y_continuous(breaks=seq(0,60,5),labels=paste0(seq(0,60,5),"%"))+
  scale_x_continuous(breaks=seq(0,60,5),labels=paste0(seq(0,60,5),"%"))+
  labs(y="Fentanyl and Fluorofentanyl",x=c.drug,title=paste0("F) ",c.drug))

gg3g <- ggMarginal(gg, type = "density",fill=c.col,alpha=0.7)
gg3g

#"phenethyl_4_anpp_percent"
c.col <- "#b2df8a"
c.drug <- "Phenethyl-4ANPP"

gg <- ggplot(conc.l[variable=="phenethyl_4_anpp_percent"],aes(y=fent_fluoro_sum,x=value))+aesth+
  geom_smooth(method='lm',alpha=.3,fill=c.col,color=c.col)+
  #facet_wrap(~variable,scales='free')+
  geom_point(shape=21,stroke=1.1,,size=2,color="black",fill=c.col)+
  theme(legend.position = 'none')+
  scale_y_continuous(breaks=seq(0,60,5),labels=paste0(seq(0,60,5),"%"))+
  scale_x_continuous(breaks=seq(0,60,1),labels=paste0(seq(0,60,1),"%"))+
  labs(y="Fentanyl and Fluorofentanyl",x=c.drug,title=c.drug)

gg3h <- ggMarginal(gg, type = "density",fill=c.col,alpha=0.7)
gg3h

pdf(paste0(root,"visuals/figure_2_",Sys.Date(),".pdf"),width=16,height=8)
grid.arrange(gg3a,gg3b,gg3c,gg3d,gg3f,gg3g,nrow=2)
dev.off()

#coprevalence of other substances by fent cat
f3 <- dt[!is.na(ffcat)]
f3 <- f3[,c("ffcat",names(f3)[names(f3)%like%"_dumb"]),with=F]
f3 <- melt.data.table(f3,id.vars = "ffcat")
f3[,variable:=str_to_title(str_trim(str_to_title(str_replace_all(variable,"_dumb|_"," "))))]
f3 <- merge(f3,c.drugs,by='variable')
f3 <- f3[Class%in%c("Fentanyl Analog","Non-Fentanyl Drug","Bulking Agent")]
f3 <- f3[variable!="Fentanyl"]
f3[,obs:=1]
fig3 <- f3[,.(prop=mean(value),n=sum(obs)),by=.(ffcat,variable,Class)]
fig3[,se:=sqrt((prop*(1-prop))/n)]
fig3[,prop_lower:=prop - 1.96*se]
fig3[,prop_upper:=prop + 1.96*se]
fig3[prop_lower<0,prop_lower:=0]
fig3[prop_upper>1,prop_upper:=1]
fig3[prop==0,prop_upper:=NA]
fig3[prop==0,prop_lower:=NA]

fig3[,ffcat:=factor(ffcat,levels=c("<1%","1-5%","5-15%",">15%"))]

ord <- fig3[ffcat=="<1%"]
ord <- ord[order(-prop)]
fig3[,variable:=factor(variable,levels=ord$variable)]

fig3[,Class:=factor(Class,levels=c("Fentanyl Analog","Non-Fentanyl Drug","Bulking Agent"))]

gg4 <- ggplot(fig3,aes(y=prop*100,x=variable,ymin=prop_lower*100,ymax=prop_upper*100,fill=ffcat,color=ffcat))+aesth+
geom_bar(stat='identity',position=position_dodge(),alpha=.5)+
geom_errorbar(position=position_dodge())+
facet_wrap(~Class,ncol=1,scales='free')+
scale_fill_brewer("% Fentanyl and Fluorofentanyl Concentration",type="qual",palette=6)+
scale_color_brewer("% Fentanyl and Fluorofentanyl Concentration",type="qual",palette=6)+
scale_y_continuous(breaks=seq(0,100,10),labels=paste0(seq(0,100,10),"%"))+
labs(y="Proportion of Co-Prevalence",x="Substance")+
geom_vline(xintercept = seq(0.5,10.5,1),alpha=.3)

pdf(paste0(root,"visuals/figure_3_",Sys.Date(),".pdf"),width=8,height=8)
print(gg4)
dev.off()
       
#supplemental table all present substances by DART-MS
st1 <- dt[!is.na(fent_fluoro_sum)]
c.cols <- c("ffcat",names(st1)[names(st1)%like%"_dumb"])
st1 <- st1[,..c.cols]
st1[,pid:=.I]
st1 <- melt.data.table(st1,id.vars = c("pid","ffcat"))
st1.p1 <- st1[,.(prop=mean(value,na.rm=T)),by=.(variable)]
st1.p2 <- st1[,.(prop=mean(value,na.rm=T)),by=.(variable,ffcat)]
st1.p2 <- dcast.data.table(st1.p2,variable~ffcat)
st1 <- merge(st1.p1,st1.p2,by="variable")
st1 <- st1[,c("variable","prop","<1%","1-5%","5-15%",">15%")]
st1[,variable:=str_to_title(str_replace_all(variable,"_dumb|_"," "))]

st1 <- st1[order(-prop)]
names(st1) <- c("Substance","Overall Proportion","Fent: <1%","Fent: 1-5%","Fent: 5-15%","Fent: >15%")

fwrite(st1,paste0(root,"REDACTED",Sys.Date(),".csv"))






