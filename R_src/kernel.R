library(kernlab)
library(Rstem)
library(Snowball) 

#-------------------kernlab package-----------------#

#RBF Kernel
rbfkernel <- rbfdot(sigma = 0.1)
#Create the kernel matrix
kernmat <- kernelMatrix(kernel=rbfkernel,x=as.matrix(dtm2))

#String Kernel
sk <- stringdot(type="string",length=3)
class(sk) <- "kernel" #This important!
kernstring <- kernelMatrix(kernel=sk,x=as.matrix(dtm2))

#Spectrum Kernel
sk2 <- stringdot(type="spectrum", length=2, normalized=FALSE)
Compute the kernel between two words
#sk2('sunny','happy')

