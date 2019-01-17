rm(list=ls()) 
setwd("C:/Rdata/")

data <- data.frame(read.csv("Diabetes_data_zaR.csv", header=T, na.strings=c("","NA"), stringsAsFactors=FALSE, sep=";"))

# --------------------------------------------------------------------------------
# -------------- deskriptivna statistika -----------------------------------------
# --------------------------------------------------------------------------------
head(data, 6)
colnames(data)
summary(data) 
descriptive_dataframe <- summary(data)   
write.table(descriptive_dataframe, file = "Diabetes_summary.csv", append = FALSE, quote=TRUE, eol = "\n", na="NA", row.names=FALSE, col.names=TRUE, 
            sep = ";")
hist(data$Presence.of.diabetes, col="green", breaks=2, xlim=c(0,1))
# u sluèaju velkog broja ulaznih varijabli preporuka je napraviti redukciju varijabli nekom od metoda
# npr. na temelju korelacija, hi-kvadrata, AIC, ili dr.metodom

# ------------------------------------------------------------------------------------------------
# -------------- Neuronska mreza za klasifikaciju ------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ----------- 1. korak - skaliranje (normalizacija) --------------------------------------------------------------
# ----------- Skaliranjem se sve vrijednosti preraèunavaju na interval (max-min)
# ----------- razlog skaliranja - kako NM ne bi radila s velikim rasponima vrijednosti

maxs <- apply(data, 2, max) ## parametar 2 oznacava da ce funkcija apply primijeniti na stupce, a ne na retke
mins <- apply(data, 2, min) ## parametar 2 oznacava da ce funkcija apply primijeniti na stupce, a ne na retke

total_sample_scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))


## ------------ 2. korak - podjela na poduzorke za treniranje i testiranje 
# ------------------------------------------------------------------------------------------------

no_rows = nrow(data)
no_rows
no_train = round(no_rows * 0.8)
no_train
no_test = no_rows - (no_train)
no_test

train_sample_scaled <- total_sample_scaled[sample(1:nrow(total_sample_scaled), no_train, replace=FALSE),]
test_sample_scaled <- total_sample_scaled[sample(1:nrow(total_sample_scaled), no_test, replace=FALSE),]

## ------------ 3. korak - Nedostajuæe vrijednosti -------------------------------------------------------------- 
## ---------- provjera ima li missing, ako nema, ispod svake var ce ispisati nulu. 
## ---------- Ako ima, tada treba zamijeniti nedostajuce s nekom metodom ----------------------------------------

apply(data,2,function(x) sum(is.na(x)))

## ----------- 4. korak - kreiranje posebnih oznaka kategorija ako se radi o multiclass problemu klasifikacije  ----------------------
## ------------ Ako se radi o binarnom problemu, 0 i 1, tada nije potrebno ---------------------------------------
# u dataframe-u je dodan stupac y za kategorijalnu izlaznu varijablu : y = train_sample$Presence.of.diabetes
#train_sample$y=ifelse(train_sample$Presence.of.diabetes,"poz","neg")
#train_sample$poz=train_sample$Presence.of.diabetes==1
#train_sample$neg=train_sample$Presence.of.diabetes!=1



## ------------ 5. korak - strukturiranje jednadžbe neuronske mreže ---------------------------------------

## ---------- strukturiranje jednadzbe za ANN model (izlazna var i ulazne) -----------------------------
feats <- names(train_sample_scaled[1:ncol(train_sample_scaled)-1])
feats
f <- paste(feats,collapse=' + ')
f <- paste('Presence.of.diabetes ~',f)
# Convert to formula
f <- as.formula(f)  ### jednadzba je sad spremljena u f i moze se tako pozivati
f
## ------------ 6. korak - pokretanje neuronske mreže ---------------------------------------

# load libs
install.packages("neuralnet")
library(neuralnet)
library(nnet)
# --------- mreza 1 - sa logistic funkcijom -------------------------------------------------
set.seed(5)
net1 <- neuralnet(f, train_sample_scaled, 
                         hidden=3, 
                        learningrate=0.01,
                        stepmax = 1e+05,
                        algorithm = "rprop+",
                        lifesign="full",
                        act.fct = "logistic",
                        err.fct = "ce", 
                        linear.output = FALSE) # linear.output mora biti na FALSE ako se radi klasifikacijaa
print(net1)

summary(net1)
plot(net1, rep="best")
plot(net1, fontsize = 9, dimension = 6, show.weights = TRUE, col.hidden.synapse="black",
     radius = 0.10, 
     arrow.length = 0.2, 
     intercept = TRUE, 
     intercept.factor = 0.4, 
     information = TRUE,
     information.pos = 0.1,
     x.entry = NULL, x.out = NULL)


## --------- Compute predictions on test sample za net 1 ----------------------------------

predict <- compute(net1, test_sample_scaled[,1:8])
predict.values = 1*(predict$net.result>0.5)
predict.values
# prints real output values of test subsample
test_sample_scaled$Presence.of.diabetes
conf.mat.test=table(test_sample_scaled$Presence.of.diabetes,predict.values[,1])
# computes hit rate on test set
conf.mat.test
net1.test.hit_rate=sum(diag(conf.mat.test))/sum(conf.mat.test)
net1.test.hit_rate
# write out results of test prediction in file
data_real_predicted <- data.frame(test_sample_scaled$Presence.of.diabetes, predict.values[,1])
colnames(data_real_predicted)<-c("Real", "Predicted")
write.table(data_real_predicted,file="net1_test_pred_values.csv",sep=";")

