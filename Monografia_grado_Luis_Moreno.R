# Script monograf�a de grado
#Comparaci�n de dos t�cnicas de clasificaci�n del gasto en la atenci�n de
#pacientes con c�ncer de mama 
# Luis Octavio Moreno Carvajal 

#Cargamos los paquetes requeridos-----------------------------------------------

library(randomForest)
library(magrittr)
library(tidyverse)
library(fastDummies)
library(VIM)
library(knitr); 
library(kableExtra)
library(mice)
library(rpart)
library(rpart.plot)
library(mod)
library(modelsummary)
library(flextable)
library(ggplot2)
library(strucchange)
library(car)


#Cargamos los datos-------------------------------------------------------------

datos <- read.csv("D:\\SEMINARIO DE GRADO\\Base_datos_grado\\CAMAMA_FINAL.csv")
datos[datos=="#N/A"] <- NA

datos  %>% head() %>% 
  kable(digits = 6,caption = "Encabezado de datos") %>%  
  kable_styling(font_size = 10)

# La base de datos tiene `3,300` observaciones y 5 variables 

#Verificamos las variables con datos faltantes ---------------------------------

aggr(datos,numbers=T,sortVar=T)

#Las variables que contienen valores faltantes son **Estadio_** y **HER2

#Efectuamos la t�cnica de imputaci�n -------------------------------------------

datos <- datos%>% mutate_if(is.character, as.factor)
Facturado_ <-datos$Facturado_
X_final<- datos[c(1,2,3,4)]
CAmama<- rfImpute(X_final, Facturado_, iter=5, ntree=50)

#Creamos las variables Dummies y verificamos la salida--------------------------

fastDummies::dummy_cols(datos, select_columns =c("Estadio_","HER2_","RH_")) %>% 
  select(-Estadio_, -HER2_,-RH_) -> datos

str(datos)

#Efectuamos un an�lisis gr�fico descriptivo ------------------------------------

datos %>% ggplot() + geom_density(aes(Edad_, color = y)) + 
  ggtitle("Gr�fico de Edad") +
  theme_classic() +scale_color_manual(values=c("gray7", "lavenderblush4"))-> p1

datos %>% ggplot() + geom_bar(aes(RH_, fill = "y")) + ggtitle("Gr�fico de RH") +
  theme_classic() +scale_fill_manual(values=c("gray7", "lavenderblush4"))-> p2

datos %>% ggplot() + geom_bar(aes(HER2_, fill = y)) + 
  ggtitle("Gr�fico para HER2") +
  theme_classic() +scale_fill_manual(values=c("gray7", "lavenderblush4")) + 
  theme(axis.text.x = element_text(size=rel(0.8), angle = 0))->p3

datos %>% ggplot() + geom_bar(aes(Estadio_, fill = y)) + 
  ggtitle("Gr�fico para Estadio") +
  theme_classic() +scale_fill_manual(values=c("gray7", "lavenderblush4"))+ 
  theme(axis.text.x = element_text(size=rel(0.8), angle =0))-> p4

gridExtra::grid.arrange(p1,p2,p3,p4,ncol = 2)


#Clasificamos la variable respuesta --------------------------------------------

datos <- CAmama %>% mutate(y =ifelse(Facturado_ <=22456373,"Bajo","Alto")) %>% 
  select(-Facturado_)

#Creamos una subset de entrenamiento y prueba para el modelo log�stico ---------

n <- nrow(datos)
set.seed(74)
datos <- datos %>% mutate(y = ifelse(y =="Bajo",0,1)) 
train <- sample(1:n, round(n*.8))
datos_entreno <- datos[train,]
datos_prueba <- datos[-train,]

#Ajustamos el modelo log�stico -------------------------------------------------

modelo_log <- glm(y ~.,data = datos_entreno,
                  family = binomial(link = "logit"))
summary(modelo_log)
modelsummary(modelo_log)
modelsummary(list(modelo_log), output = "table.docx")
glm.probs<-predict(modelo_log, type="response")
glm.probs[1:10]


#Efectuamos la predicci�n con el modelo log�stico ------------------------------

fit_log <- round(predict(modelo_log, newdata = datos_entreno, type ="response"))
pred_log <- round(predict(modelo_log, newdata = datos_prueba, type ="response"))

#Calculamos los errores --------------------------------------------------------

error_entreno <- 100*mean(datos_entreno$y == fit_log)
error_prueba <-  100*mean(datos_prueba$y == pred_log)
data.frame(error_entreno, error_prueba) -> error
colnames(error) <- c("Entreno", "Prueba")
error %>% 
  kable(digits = 6,caption = "Score Modelo Logistico") %>%  
  kable_styling(font_size =15)

#Calculamos la matriz de confusion para el modelo log�stico --------------------

d <- table(datos_prueba$y, pred_log)
colnames(d) <- c("Bajo Pred", "Alto Pred")
rownames(d) <- c("Bajo Real", "Alto Real")
d %>% kable(digits = 6,caption = "Matriz de confuci�n") %>%  
  kable_styling(font_size =15)

# Creamos una subset de entrenamiento y prueba para el �rbol de decisi�n -------

set.seed(74)
datos$y <- as.factor(datos$y)
train <- sample(1:n, round(n*.8))
datos_entreno <- datos[train,]
datos_prueba <- datos[-train,]

#Ajustamos el modelo de �rbol de decisi�n---------------------------------------

mod_arb <- rpart(y ~ Edad_ +Estadio_ + HER2_ + RH_, data = datos_entreno)
mod_arb
rpart.plot(mod_arb, type = 3, digits = 4, roundint = TRUE)

#Efectuamos la predicci�n con los �rboles decisi�n -----------------------------

fit_arb <- predict(mod_arb, newdata = datos_entreno, type = "class")
pred_arb <- predict(mod_arb, newdata = datos_prueba, type = "class")

#Calculamos los errores --------------------------------------------------------

error_entreno <- 100*mean(datos_entreno$y == fit_arb)
error_prueba <-  100*mean(datos_prueba$y == pred_arb)
data.frame(error_entreno, error_prueba) -> error
colnames(error) <- c("Entreno", "Prueba")
error %>% 
  kable(digits = 6,caption = "Score Modelo de arboles") %>%  
  kable_styling(font_size = 15)

#Calculamos la matriz de confusion para el �rbol -------------------------------

a <- table(datos_prueba$y, pred_arb)
colnames(a) <- c("Bajo Pred", "Alto Pred")
rownames(a) <- c("Bajo Real", "Alto Real")
a %>% kable(digits = 6,caption = "Matriz de confusi�n") %>%  
  kable_styling(font_size =15)

#----------------------FIN------------------------------------------------------
