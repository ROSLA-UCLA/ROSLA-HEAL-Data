#code for: "What Does 'Polysubstance' Really Mean? Comparing Drug-Involved Deaths in Los Angeles in CDC Records vs. Detailed Medical Examiner Data"  

## Setup
rm(list = ls())
pacman::p_load(data.table, tidyverse, ggplot2,ggrepel, grid, gridExtra,lubridate,cowplot,stringr,forecast,purrr,ggridges)
root <- "SET_WD"
aesth <- theme_bw() + theme(axis.title = element_text(size=10,face='bold'),axis.text =element_text(size=10,face='bold'),plot.title =element_text(size=12,face='bold'),strip.background = element_rect(fill="white"),strip.text=element_text(size=10,face='bold'),legend.position = 'top')
aesth_classic <- theme_classic() + theme(axis.title = element_text(size=10,face='bold'),axis.text =element_text(size=10,face='bold'),plot.title =element_text(size=12,face='bold'),strip.background = element_rect(fill="white"),strip.text=element_text(size=10,face='bold'),legend.position = 'top')

#read LA ME data
dt <- fread(paste0(root,"REDACTED.csv"))
dt[,pid:=.I]
dt[,Fentanyl:=as.numeric(Fentanyl)]
dt[,Alcohol:=as.numeric(Alcohol)]
dt[,Others:=as.numeric(Others)]

#add date info
dt[,date:=as.Date(DeathDate)]
dt[,year:=year(date)]

#add xylazine  
dt[,Xylazine:=0]
dt[grepl("XYLAZINE",CauseA)|grepl("XYLAZINE",CauseB),Xylazine:=1]

#total number of deaths involved in each death, over time
dt.drugs <- dt[,c("pid","year","Methamphetamine","Heroin","Cocaine","Fentanyl","Alcohol","Prescription.opioids","Any Opioids","Xylazine","Benzodiazepines","Others")]
dt.d.l <- melt.data.table(dt.drugs,id.vars = c("pid","year"))
dt.d.l[,value:=as.numeric(value)]
dt.d.y <- dt.d.l[,.(num=sum(value)),by=.(pid,year)]
dt.d.y[num==2,polycapt:=1]
dt.d.y[num>2,polycapt:=0]

dt.d.y.summ <- dt.d.y[,.(av_num=mean(num),prop_polycapt=mean(polycapt,na.rm=T)),by=.(year)]
dt.d.y[,year:=factor(year,levels=seq(2012,2024,1))]

#figure 1
gg1a <- ggplot(dt.d.y,aes(x=num,y=factor(year),fill=factor(year),group=factor(year)))+aesth+
  geom_density_ridges(bandwitdth=1.0)+theme(legend.position = 'none')+
  geom_vline(aes(xintercept=2.5,alpha=0.4))+
  geom_vline(aes(xintercept=1.5,alpha=0.4))+
  scale_x_continuous(breaks=seq(1,8,1),limits = c(0.5,5.2))+
  scale_y_discrete(limits = rev(unique(sort(dt.d.y$year))))+
   labs(y="Year",x="Number of Involved Substances",title="A) Drug-Involved Deaths by Number of Substances Involved")

gg1c <- ggplot(dt.d.y.summ,aes(x=factor(year),y=(1-prop_polycapt)*100,fill=(1-prop_polycapt)*100))+aesth+
geom_bar(stat='identity',color='grey')+
labs(title="C) Percent of Polysubstance Deaths Incompletely Categorized\n by 2 Drug Descriptors",
y="Percent",x="Year")+
scale_y_continuous(breaks = seq(0,70,10),labels=paste0(seq(0,70,10),"%"))+
geom_label(aes(y=((((1-prop_polycapt)*100))+4),label=paste0(round((1-prop_polycapt)*100),"%")))+
scale_fill_gradient(low='white',high="#1f78b4",guide = guide_colourbar("",binwidth=2.0))+
  theme(legend.position = 'none')


gg1b <- ggplot(dt.d.y.summ,aes(x=factor(year),y=av_num,fill=av_num))+aesth+
  geom_bar(stat='identity',color='grey')+
  labs(title="B) Average Number of Drugs Present per Drug-Involved Death",
       y="Percent",x="Year")+
  #scale_y_continuous(breaks = seq(0,70,10),labels=paste0(seq(0,70,10),"%"))+
  geom_label(aes(y=av_num,label=(round(av_num,2))))+
  scale_fill_gradient(low='white',high="#33a02c",guide = guide_colourbar("",binwidth=2.0))+
  theme(legend.position = 'none')

pdf(paste0(root,"visuals/polysubstance_",Sys.Date(),".pdf"),height=6,width=12)
grid.arrange(gg1a,gg1b,gg1c,layout_matrix = rbind(c(1, 2),
                                                  c(1, 3)))
dev.off()

#Figure 2
dt.drugs[,obs:=1]

combs <- dt.drugs[,.(obs=sum(obs)),
by=.(year,Methamphetamine,Cocaine,Fentanyl,Heroin,Prescription.opioids,Xylazine,Alcohol,Benzodiazepines,Others)]
combs[is.na(combs)]<-0

#rank clusters by year
combs[,num:=as.character(Methamphetamine+Cocaine+Fentanyl+Heroin+Prescription.opioids+Xylazine+Alcohol+Benzodiazepines+Others)]
combs[,cid:=paste0(year,Methamphetamine,Cocaine,Fentanyl,Heroin,Prescription.opioids,Xylazine,Alcohol,Benzodiazepines,Others)]

combs[,tot:=sum(obs),by=.(year)]
combs[,perc:=obs/tot]
combs[,rank:=frank(perc,ties.method = 'first'),by=.(year)]
combs[,max_rank:=max(rank),by=.(year)]

combs <- combs[order(year,-perc,num,Methamphetamine,Cocaine,Fentanyl,Heroin,Prescription.opioids,Xylazine,Alcohol,Benzodiazepines,Others)]
combs[,cid:=factor(cid,levels=unique(combs$cid))]

#dot component
gg2b <- ggplot(combs[num>2&year%in%c(2012,2018,2024)]) + aesth+
  geom_point(size=5,aes(y=9,x=cid,color=ifelse(Methamphetamine==1,num,"Not\nPresent"),alpha=ifelse(Methamphetamine==1,.95,.05))) +
  geom_point(size=5,aes(y=8,x=cid,color=ifelse(Cocaine==1,num,"Not\nPresent"),alpha=ifelse(Cocaine==1,.95,.05))) +
  geom_point(size=5,aes(y=7,x=cid,color=ifelse(Fentanyl==1,num,"Not\nPresent"),alpha=ifelse(Fentanyl==1,.95,.05))) +
  geom_point(size=5,aes(y=6,x=cid,color=ifelse(Heroin==1,num,"Not\nPresent"),alpha=ifelse(Heroin==1,.95,.05))) +
  geom_point(size=5,aes(y=5,x=cid,color=ifelse(Prescription.opioids==1,num,"Not\nPresent"),alpha=ifelse(Prescription.opioids==1,.95,.05))) +
  geom_point(size=5,aes(y=4,x=cid,color=ifelse(Xylazine,num,"Not\nPresent"),alpha=ifelse(Xylazine==1,.95,.05))) +
  geom_point(size=5,aes(y=3,x=cid,color=ifelse(Alcohol==1,num,"Not\nPresent"),alpha=ifelse(Alcohol==1,.95,.05))) +
  geom_point(size=5,aes(y=2,x=cid,color=ifelse(Benzodiazepines==1,num,"Not\nPresent"),alpha=ifelse(Benzodiazepines==1,.95,.05))) +
  geom_point(size=5,aes(y=1,x=cid,color=ifelse(Others==1,num,"Not\nPresent"),alpha=ifelse(Others==1,.95,.05))) +
  scale_color_manual(name="Drug\nNumer",values=c("3"="#440154FF","4"="#33638DFF","5"= "#29AF7FFF"))+
  theme(axis.title.x=element_blank(),axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),legend.position = "none") + guides(alpha=F)+
  scale_y_continuous(breaks=seq(1,9),
                     labels=rev(c("Methamphetamine","Cocaine","Fentanyl","Heroin","Prescription.opioids","Xylazine","Alcohol","Benzodiazepines","Others")))+
  labs(y="Drug Involved")+
  #facet_wrap(~year,strip.position="bottom",nrow=1,scales='free_x')
  facet_grid(~year,scales='free',space = "free")
  

combs[num>2&year%in%c(2012)]
combs[num>2&year%in%c(2023)]
combs[num>2&year%in%c(2024)]

combs[num==4&year%in%c(2024)]
combs[num==5&year%in%c(2024)]



#barchart component
gg2a <- ggplot(combs[num>2&year%in%c(2012,2018,2024)],aes(x=cid,y=perc*100,fill=factor(num))) + theme_classic()+
  scale_fill_manual(name="Drug\nNumer",values=c("3"="#440154FF","4"="#33638DFF","5"= "#29AF7FFF"))+
  theme(axis.title.x=element_blank(),axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),legend.position = "top",
        strip.text=element_blank()) + geom_bar(stat='identity')+
  labs(y="% of Deaths",title="Substance Clusters Observed Among Deaths Involving 3 or More Drugs") + 
  #geom_vline(xintercept=c(6.5,12.5,18.5),alpha=.9,size=1.15) +
  scale_y_continuous(breaks=seq(0,40,1),labels=paste0(seq(0,40,1),"%"),
                     expand=expansion(mult = c(.01, 0.3), add = c(.0, 0)))+
  #facet_wrap(~year,strip.position="top",nrow=1,scales='free_x')
  facet_grid(~year,scales='free',space = "free")

pdf(paste0(root,"/visuals/Polysubstance_Up_Set",Sys.Date(),".pdf"),width=16,height=4)
plot_grid(gg2a, gg2b, ncol=1, align="v",rel_heights=c(2,3),axis="lr")
dev.off()

  

#compare CDC Wonder data for LA County to ME data at year-drug level

#fent prov: https://wonder.cdc.gov/controller/saved/D176/D440F934
#meth prov: https://wonder.cdc.gov/controller/saved/D176/D440F935
#cocaine prov: https://wonder.cdc.gov/controller/saved/D176/D440F936
#heroin: https://wonder.cdc.gov/controller/saved/D176/D440F938
#rx opioids: https://wonder.cdc.gov/controller/saved/D176/D440F942
#benzos: https://wonder.cdc.gov/controller/saved/D176/D440F943

reaDrug <- function(loc,nm){
  dt <- fread(paste0(root,"data/",loc,".csv"))
  dt <- dt[Notes==""]
  dt[,Year:=`Year Code`]
  dt <- dt[,c("Year","Deaths")]
  setnames(dt,"Deaths",nm)
  return(dt)
}


all.p <- purrr::reduce(.x = list(
  reaDrug("CDC/prov/fent","Fentanyl"),
  reaDrug("CDC/prov/heroin","Heroin"),
  reaDrug("CDC/prov/rx","Prescription.opioids"),
  reaDrug("CDC/prov/cocaine","Cocaine"),
  reaDrug("CDC/prov/meth","Methamphetamine"),
  reaDrug("CDC/prov/benzo","Benzodiazepines")
), merge, by = c('Year'), all = T)

all.p.l <- melt.data.table(all.p,id.vars = c("Year"),value.name = "CDC")

dt.drugs.l <- melt.data.table(dt.drugs,id.vars = c("pid","year","obs"))
dt.drugs.l <- dt.drugs.l[value==1]

dt.drugs.l <- dt.drugs.l[,.(ME=sum(obs)),by=.(variable,year)]
dt.drugs.l[,Year:=year]

comp <- merge(dt.drugs.l,all.p.l,by=c("Year","variable"))

gg3 <- ggplot(comp,aes(y=CDC,x=ME,fill=variable,color=variable))+aesth+facet_wrap(~variable,scales='free')+
  geom_point()+
  geom_smooth(method='lm',se=F)+
  geom_abline(slope=1,intercept=0)+
  theme(legend.position = "none")+
  labs(y="CDC WONDER Data",x="Medical Examiner Data",title="Comparing Counts of Drug-Involved Deaths in CDC Wonder vs.\n LA County Medical Examiner Data, 2018-2024")

pdf(paste0(root,"/visuals/Single_substance_Compare_ME_CDC",Sys.Date(),".pdf"),width=8,height=5)
gg3
dev.off()

cor(comp$ME, comp$CDC)^2

comp[variable=="Methamphetamine"]

