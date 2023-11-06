#!/usr/bin/env R
# =============================================================================
#     File: hamiltonian.R
#  Created: 2023-11-05 21:01
#   Author: Bernie Roesler
#
# Hamiltonian Monte Carlo (HMC) simulation from Section 9.3.2. Overthinking.
# =============================================================================

library(rethinking)

# (R code 9.3) U needs to return neg-log-probability
myU4 <- function( q , a=0 , b=1 , k=0 , d=1 ) {
    muy <- q[1]
    mux <- q[2]
    U <- sum( dnorm(y,muy,1,log=TRUE) ) 
        + sum( dnorm(x,mux,1,log=TRUE) ) 
        + dnorm(muy,a,b,log=TRUE) 
        + dnorm(mux,k,d,log=TRUE)
    return( -U )
}

# (R code 9.4) gradient function
# need vector of partial derivatives of U wrt vector q
myU_grad4 <- function( q , a=0 , b=1 , k=0 , d=1 ) {
    muy = q[1]
    mux = q[2]
    G1 <- sum( y - muy ) + (a - muy)/b^2
    G2 <- sum( x - mux ) + (k - mux)/b^2
    return( c( -G1, -G2 ) )  # negative because energy is neg-log-prob
}

# test data
set.seed(7)
y <- rnorm(50)
x <- rnorm(50)
x <- as.numeric(scale(x))
y <- as.numeric(scale(y))

# (R code 9.5)
library(shape)  # for fancy arrows
Q <- list()
Q$q <- c(-0.1, 0.2)
pr <- 0.3
plot( NULL , ylab="muy" , xlab="mux" , xlim=c(-pr,pr) , ylim=c(-pr,pr) )
step <- 0.03
L <- 11  # 0.03/28 for U-turns --- 11 for working example
n_samples <- 4
path_col <- col.alpha("black",0.5)
points( Q$q[1] , Q$q[2] , pch=4 , col="black" )
for ( i in 1:n_samples ) {
    Q <- HMC2( myU4, myU_grad4 , step , L , Q$q )
    if ( n_samples < 10 ) {
        for ( j in 1:L ) {
            K0 <- sum(Q$ptraj[j,]^2)/2  # kinetic energy
            lines( Q$traj[j:(j+1),1] , Q$traj[j:(j+1),2] , col=path_col , lwd=1+2*K0 )
        }
        points( Q$traj[1:L+1,] , pch=16 , col="white" , cex=0.35 )
        Arrows( Q$traj[L,1] , Q$traj[L,2] , Q$traj[L+1,1] , Q$traj[L+1,2] ,
               arr.length=0.25 , arr.adj=0.7 )
        text( Q$traj[L+1,1] , Q$traj[L+1,2] , i , cex=0.8 , pos=4 , offset=0.4 )
    }
    points( Q$traj[L+1,1] , Q$traj[L+1,2] , pch=ifelse( Q$accept==1 , 16 , 1 ) ,
           col=ifelse( abs(Q$dH)>0.1 , "red" , "black" ) )
}

# (R code 9.6)
# HMC2 <- function ( ...

# =============================================================================
# =============================================================================
