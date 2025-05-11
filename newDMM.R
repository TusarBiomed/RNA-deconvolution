## DMM code

newDMM <- function(numSamples, colData, initGEP, design, dtype = "float64"){
  ## construct model matrix
  modelMatrix <- model.matrix(design, data = colData)
  ## Ws
  if (length(dim(initGEP)) == 2){
    ## determine the number of genes
    numGenes <- NROW(initGEP)
    ## determine the number of cell types
    numCelltypes <- NCOL(initGEP)
    numberInitialGEPs <- 1

  } else if (length(dim(initGEP)) == 3){
    ## determine the number of genes
    numGenes <- dim(initGEP)[2]
    ## determine the number of cell types
    numCelltypes <- dim(initGEP)[3]
    numberInitialGEPs <- dim(initGEP)[1]
  } else{
    stop("initGEP is neither a 2D matrix nor a tensor with 3 dimensions")
  }

  ## Ws
  W <- array(
    0,
    dim = c(NCOL(modelMatrix), numGenes, numCelltypes)
  )
  W[seq_len(numberInitialGEPs),,] = initGEP


  keras_model_custom(
    model_fn = function(self){
      ## modelMatrix
      self$modelMatrix <- tf$constant(modelMatrix, dtype = dtype)
      ## GEP
      self$W <- tf$Variable(W, dtype = dtype, trainable = TRUE, name = "W")
      ## mixing
      self$H <- tf$Variable(array(0, dim = c(numCelltypes, numSamples)), dtype = dtype, trainable = TRUE, name = "H")
      # Number of samples
      self$numSamples <- tf$constant(as.integer(numSamples))

      function(counts, mask = NULL, training = TRUE){
        H <- tf$exp(self$H)
        W <- tf$expand_dims(self$W, axis = 0L)
        W <- tf$'repeat'(W, self$numSamples, axis = 0L)
        W <- tf$einsum('pqrs,pq->pqrs', W, self$modelMatrix, name = "W")
        W <- tf$reduce_sum(W, axis = 1L)
        W <- tf$exp(W)
        alpha <- tf$einsum('prs,sp->pr', W, H, name = "alpha")
        alphaN <- counts + alpha
        tf$reduce_sum(tf$math$lbeta(alpha) - tf$math$lbeta(alphaN))
      }
    }
  )
}


newOptDMM <- function(counts, colData, initGEP, design, tol = .Machine$double.eps, maxit = 10, dtype = "float64"){
  numSamples <- NCOL(counts)
  model <- newDMM(numSamples, colData, initGEP, design)

  dims = lapply(model$get_weights(), dim)

  counts <- t(counts)
  GRAD <- NULL

  targetH <- function(p){
    weights <- model$get_weights()
    weights[[2]] = array(p, dim = dims[[2]])
    model$set_weights(weights)
    with(tf$GradientTape() %as% tape, {
      tape$watch(model$trainable_variables[[2]])
      loss_value <- model(counts)
    })
    GRAD <<- unlist(lapply(tape$gradient(loss_value, model$trainable_variables[[2]]),
                           function(x){
                             as.array(x)
                           }))
    # assign("GRAD", unlist(lapply(tape$gradient(loss_value, model$trainable_variables[[2]]),
    #                              function(x){
    #                                as.array(x)
    #                                })), envir = parent.frame())
    as.numeric(loss_value)
  }

  targetW <- function(p){
    weights <- model$get_weights()
    weights[[1]] = array(p, dim = dims[[1]])
    model$set_weights(weights)
    with(tf$GradientTape() %as% tape, {
      tape$watch(model$trainable_variables[[1]])
      loss_value <- model(counts)
    })
    GRAD <<- unlist(lapply(tape$gradient(loss_value, model$trainable_variables[[1]]),
                           function(x){
                             as.array(x)
                           }))
    as.numeric(loss_value)
  }
  gradient <- function(p){
    GRAD
  }

  h <- unlist(model$get_weights()[[2]])
  optH <- optim(
    par = h,
    fn = targetH,
    gr = gradient,
    method = "L-BFGS-B",
    control = list(trace = 1, maxit = 5000),
    lower = -20,
    upper = 15
  )
  pars <- model$get_weights()
  pars[[2]] <- array(optH$par, dim = dims[[2]])
  model$set_weights(pars)

  oldPars <- unlist(model$get_weights())
  parSave <- list()
  for (iter in seq_len(2000)){
    h <- unlist(model$get_weights()[[2]])
    optH <- optim(
      par = h,
      fn = targetH,
      gr = gradient,
      method = "L-BFGS-B",
      control = list(maxit = 100),
      lower = -20,
      upper = 15
    )
    pars <- model$get_weights()
    pars[[2]] <- array(optH$par, dim = dims[[2]])
    model$set_weights(pars)
    w <- unlist(model$get_weights()[[1]])
    optW <- optim(
      par = w,
      fn = targetW,
      gr = gradient,
      method = "L-BFGS-B",
      control = list(maxit = 100),
      lower = -20,
      upper = 20
    )
    pars <- model$get_weights()
    pars[[1]] <- array(optW$par, dim = dims[[1]])
    model$set_weights(pars)
    currentPars <- unlist(model$get_weights())
    cat(iter, " - ", mean(abs(currentPars - oldPars)), optW$value, "\n")

    # cat(iter,"th iteration has finished \n")
    parSave[[iter]] = model$get_weights()
    if (all(abs(currentPars - oldPars) < tol)){ #  || (optW$convergence == 0 && optH$convergence == 0)
      cat(all(abs(currentPars - oldPars)), "tol = ", tol, "h = ", optH$convergence, "w = ", optW$convergence, "\n")
      cat("Model reached the optimum point \n")
      break
    }
    oldPars <- currentPars
  }
  w <- unlist(model$get_weights()[[1]])
  optW <- optim(
    par = w,
    fn = targetW,
    gr = gradient,
    method = "L-BFGS-B",
    control = list(trace = 1, maxit = 5000),
    lower = -20,
    upper = 20
  )
  pars <- model$get_weights()
  pars[[1]] <- array(optW$par, dim = dims[[1]])

  pars
}

######

dIn <- function(mod, countDa){

  loss <- function(design, counts){
    design(counts)
  }
  countDa <- tf$constant(t(countDa), dtype = "float64")
  with(tf$GradientTape() %as% tape1, {
    with(tf$GradientTape() %as% tape2, {
      loss_value <- loss(mod, countDa)
    })
    grad = tape2$gradient(loss_value, mod$trainable_variables)
  })

  dd <- lapply(tape1$gradient(grad, mod$trainable_variables), as.array)

  lapply(
    dd,
    function(x){
      res = sqrt(1/x)
      res[!is.finite(res)] = NA
      res
    })
}


## Function to obtain differential gene expression result

lfcResults <- function(estimates, der, gNames) {
  lapply(
    seq(dim(estimates[[1]])[1] - 1),
    function(j) {
      lapply(
        seq(dim(estimates[[2]])[1]),
        function(i){
          stat = estimates[[1]][1 + j, , i] / der[[1]][1+j, , i]
          pvals = pnorm(-abs(stat)) * 2
          log2FC = estimates[[1]][1+j, , i] / log(2)
          lfcse = der[[1]][1+j, , i]
          padj = p.adjust(pvals, method = "fdr")
          data.frame(genes = gNames,
                     L2FC = log2FC,
                     lfcse = lfcse,
                     stat = stat,
                     pvalue = pvals,
                     padj = padj)
        })
    }
  )
}


## Cell type proportion estimation

cellTP <- function(estimates) {
  if(nargs() > 1){stop("cellTP function requires at most 1 argument.")}

  apply(
    estimates[[2]],
    2,
    function(sample){
      maxSamp <- max(sample)
      z <- maxSamp + log(sum(exp(sample - maxSamp)))
      exp(sample - z) * 100
    })
}


#################
# Plotting the trace


newOptDMM <- function(counts, colData, initGEP, design, tol = .Machine$double.eps, maxit = 10, dtype = "float64"){
  numSamples <- NCOL(counts)
  model <- newDMM(numSamples, colData, initGEP, design)

  dims = lapply(model$get_weights(), dim)

  counts <- t(counts)
  GRAD <- NULL

  targetH <- function(p){
    weights <- model$get_weights()
    weights[[2]] = array(p, dim = dims[[2]])
    model$set_weights(weights)
    with(tf$GradientTape() %as% tape, {
      tape$watch(model$trainable_variables[[2]])
      loss_value <- model(counts)
    })
    GRAD <<- unlist(lapply(tape$gradient(loss_value, model$trainable_variables[[2]]),
                           function(x){
                             as.array(x)
                           }))
    as.numeric(loss_value)
  }

  targetW <- function(p){
    weights <- model$get_weights()
    weights[[1]] = array(p, dim = dims[[1]])
    model$set_weights(weights)
    with(tf$GradientTape() %as% tape, {
      tape$watch(model$trainable_variables[[1]])
      loss_value <- model(counts)
    })
    GRAD <<- unlist(lapply(tape$gradient(loss_value, model$trainable_variables[[1]]),
                           function(x){
                             as.array(x)
                           }))
    as.numeric(loss_value)
  }
  gradient <- function(p){
    GRAD
  }

  h <- unlist(model$get_weights()[[2]])
  optH <- optim(
    par = h,
    fn = targetH,
    gr = gradient,
    method = "L-BFGS-B",
    control = list(trace = 1, maxit = 1000),
    lower = -15,
    upper = 20
  )
  pars <- model$get_weights()
  pars[[2]] <- array(optH$par, dim = dims[[2]])
  model$set_weights(pars)

  oldPars <- unlist(model$get_weights())
  parSave <- list()
  iters <- list()
  vals <- list()
  for (iter in seq_len(2000)){
    h <- unlist(model$get_weights()[[2]])
    optH <- optim(
      par = h,
      fn = targetH,
      gr = gradient,
      method = "L-BFGS-B",
      control = list(maxit = 100),
      lower = -15,
      upper = 20
    )
    pars <- model$get_weights()
    pars[[2]] <- array(optH$par, dim = dims[[2]])
    model$set_weights(pars)
    w <- unlist(model$get_weights()[[1]])
    optW <- optim(
      par = w,
      fn = targetW,
      gr = gradient,
      method = "L-BFGS-B",
      control = list(maxit = 100),
      lower = -20,
      upper = 20
    )
    pars <- model$get_weights()
    pars[[1]] <- array(optW$par, dim = dims[[1]])
    model$set_weights(pars)
    currentPars <- unlist(model$get_weights())
    cat(iter, " - ", mean(abs(currentPars - oldPars)), optW$value, "\n")
    iters[[iter]] <- iter
    vals[[iter]] <- optW$value
    plot(iters, vals, type = "b", pch = 16, col = "blue",
         xlab = "Iteration", ylab = "Parameters value difference",
         main = "Progress of optimizytion")

    parSave[[iter]] = model$get_weights()
    if (all(abs(currentPars - oldPars) < tol)){ #  || (optW$convergence == 0 && optH$convergence == 0)
      cat(all(abs(currentPars - oldPars)), "tol = ", tol, "h = ", optH$convergence, "w = ", optW$convergence, "\n")
      cat("Model reached the optimum point \n")
      break
    }
    oldPars <- currentPars
  }
  w <- unlist(model$get_weights()[[1]])
  optW <- optim(
    par = w,
    fn = targetW,
    gr = gradient,
    method = "L-BFGS-B",
    control = list(trace = 1, maxit = 5000),
    lower = -20,
    upper = 20
  )
  pars <- model$get_weights()
  pars[[1]] <- array(optW$par, dim = dims[[1]])

  pars
}



