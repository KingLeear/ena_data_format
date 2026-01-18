library(jsonlite)


args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- args_all[grep("^--file=", args_all)]
this_file <- sub("^--file=", "", file_arg[1])
this_dir <- dirname(normalizePath(this_file))

source(file.path(this_dir, "ena_functions.R"))
cat("DEBUG: sourced ena_functions.R from:", file.path(this_dir, "ena_functions.R"), "\n")


args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]
out_rdata <- args[2]
config_path <- args[3]

cfg <- fromJSON(config_path)

build_ena_set(
  csv_path = csv_path,
  
  out_rdata = out_rdata,
  unitCols = cfg$unitCols,
  codesExclude = cfg$codesExclude,
  conversationCols = cfg$conversationCols,
  groupsVar = cfg$groupsVar,
  groups = cfg$groups,
  model = cfg$model,
  window.size.back = cfg$`window.size.back`,
  weight.by = cfg$`weight.by`,
  mean = cfg$mean,
  points = cfg$points,
  include.plots = FALSE,
  print.plots = FALSE,
  object_name = cfg$object_name
)

cat("OK: wrote ", out_rdata, "\n")