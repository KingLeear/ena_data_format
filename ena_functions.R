library(rENA)

build_ena_set <- function(
  csv_path,
  out_rdata,
  unitCols,
  codesExclude,
  conversationCols,
  groupsVar,
  groups,
  model = "EndPoint",
  window.size.back = 2,
  weight.by = "$",
  mean = TRUE,
  points = TRUE,
  include.plots = TRUE,
  print.plots = TRUE,
  object_name = "set.ena",
  name_mode = c("strict", "normalize")   # 
) {
  name_mode <- match.arg(name_mode)

  file <- read.csv(
    csv_path,
    fileEncoding = "UTF-8-BOM",
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  if (name_mode == "normalize") {
    colnames(file) <- make.names(colnames(file), unique = TRUE)
    unitCols <- make.names(unitCols)
    conversationCols <- make.names(conversationCols)
    if (!is.null(groupsVar) && nzchar(groupsVar)) groupsVar <- make.names(groupsVar)
    if (!is.null(codesExclude)) codesExclude <- make.names(codesExclude)
  }

  headers <- colnames(file)

  missing_units <- setdiff(unitCols, headers)
  if (length(missing_units) > 0) {
    stop(sprintf(
      "unitCols not found in data: %s\nAvailable columns: %s",
      paste(missing_units, collapse = ", "),
      paste(headers, collapse = ", ")
    ))
  }

  missing_conv <- setdiff(conversationCols, headers)
  if (length(missing_conv) > 0) {
    stop(sprintf(
      "conversationCols not found in data: %s\nAvailable columns: %s",
      paste(missing_conv, collapse = ", "),
      paste(headers, collapse = ", ")
    ))
  }

  if (!is.null(groupsVar) && nzchar(groupsVar) && !(groupsVar %in% headers)) {
    stop(sprintf(
      "groupsVar '%s' not found in data.\nAvailable columns: %s",
      groupsVar,
      paste(headers, collapse = ", ")
    ))
  }

  codesExclude <- intersect(codesExclude, headers)

  codesCols <- setdiff(headers, codesExclude)

  

  if (length(codesCols) == 0) {
    stop("codesCols is empty after excluding columns. Please adjust codesExclude.")
  }

  ena_obj <- ena(
    data = file,
    units = unitCols,
    codes = codesCols,
    model = model,
    metadata = NULL,
    weight.by = weight.by,
    conversation = conversationCols,
    window.size.back = window.size.back,
    mean = mean,
    points = points,
    include.plots = include.plots,
    print.plots = print.plots,
    groupVar = groupsVar,
    groups = groups
  )
  
  ena_obj$`_function.params` <- list(
    groupVar = groupsVar,
    unit.by  = if (length(unitCols) > 0) unitCols[length(unitCols)] else NULL
  )
  assign(object_name, ena_obj)
  save(list = object_name, file = out_rdata, compress = "xz")

  return(out_rdata)
}