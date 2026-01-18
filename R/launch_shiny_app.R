library(shiny)

args <- commandArgs(trailingOnly = TRUE)
app_dir <- args[1]
port <- as.integer(args[2])

runApp(app_dir, port = port, launch.browser = FALSE, host = "127.0.0.1")