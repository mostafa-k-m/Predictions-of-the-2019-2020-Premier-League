library(shiny)
library(shinydashboard)
library(shinydashboardPlus)
library(DT)
library(tidyverse)
library(shinyjs)
library(plotly)
library(reticulate)
source_python("learn.py")

ui <- shinyUI(

    dashboardPagePlus(skin="red",
                  dashboardHeaderPlus(title = "2019/2020 Premier League Predictions",
                                  titleWidth = 350),

                  dashboardSidebar(
                      sidebarMenu(
                          menuItem(text = "About", tabName = "about", icon=icon("clipboard")),
                          menuItem("League Predictions", tabName = "predictions", icon=icon("laptop-code"))
                      )
                  ),


                  dashboardBody(
                      tabItems(
                          tabItem(tabName = "about", h2("Predictions of  2019/2020 the Premier League Table"),
                                               h4("Final project for NU course: CIT-651: Introduction to Machine Learning & Statiscal Analysis"),br(),h3("Prepared by:"),
                                  h4("Eslam Hossam El Emam - 191084."),h4("Hossam Ragab Abdullah - 191007."),h4("Mohamed Youssef Nada - 191014."),h4("Mostafa Kamal Mostafa - 191013."),br(),br(),h4("Please press the button below to run the model on the latest available. After that please start reviewing the results from the tabs on the right."),br(),br(),
                                  fluidRow(column(12, align="center",
                                           uiOutput('updateUI'),br())),
                                  fluidRow(dataTableOutput('readytable'))),
                          tabItem(tabName = "predictions",
                                fluidRow(
                                  box(title = "Predicted Premier League Table"
                                      , status = "danger", solidHeader = T
                                      , collapsible = T, width = 8
                                      , column( 12,align="center" ,dataTableOutput('resultstable'))),

                                  box(title = "ROC curve",id="tabchart1",
                                                  tabPanel("a",plotlyOutput("plot1")),width = 4),
                                  infoBoxOutput("value1", width = 4),
                                  box(title = "Scatter Plot After PCA",id="tabchart2",
                                                  tabPanel("s",plotlyOutput("plot2")),width = 4)
                                                )

                          )

                                           )

                      )
                  )
    )


server <- shinyServer(function(input, output, session){

    controlVar <- reactiveValues(updateReady = FALSE, resultsReady = FALSE)

    dat <- NULL
    df <- read_csv("final_data.csv")

    output$updateUI <- renderUI({
        actionButton('update','Predict On Live Data', icon("gears"),
    style="color: #fff; background-color: #659e49; border-color: #2e6da4")
    })

    output$value1 <- renderInfoBox({
            simple_auc <- function(TPR, FPR){
        # inputs already sorted, best scores first
        dFPR <- c(diff(FPR), 0)
        dTPR <- c(diff(TPR), 0)
        sum(TPR * dFPR) + sum(dTPR * dFPR)/2
      }
      roc <- read_csv("reticulate1.csv")
        infoBox(title = "ROC curve AUC=",
                value = with(roc, simple_auc(TPR, FPR)),
                icon = icon("calculator"),
                fill = TRUE,
                color = "yellow")

    })
    observeEvent(input$update, {
      controlVar$updateReady <- TRUE
      temp <- tempfile()
      download.file("https://projects.fivethirtyeight.com/data-webpage-data/datasets/soccer-spi.zip",temp)
      df2 <- read_csv(unz(temp, "soccer-spi/spi_matches.csv"))
      unlink(temp)

      df2 <- df2 %>% filter(league == "Barclays Premier League")
      number_of_matches <- df2 %>% filter(!is.na(score1)) %>% nrow()

      df3 <- df2[1:(df %>% filter(!is.na(importance1)) %>% nrow()),] %>% mutate(importance1 = if_else(is.na(importance1), 0, importance1))

      df3 <- df3[1:(df %>% filter(!is.na(importance1)) %>% nrow()),] %>% mutate(importance2 = if_else(is.na(importance2), 0, importance2))
      new <- df3 %>%
        slice(0:1520) %>%
        mutate(xG1 = xg1 + nsxg1, xG2 = xg2 + nsxg2) %>%
        select(date, team1, team2, team2, spi1, spi2, importance1, importance2, xG1, xG2, score1, score2)

      df$spi1[1319:number_of_matches] <- new$spi1[1319:number_of_matches]
      df$spi2[1319:number_of_matches] <- new$spi2[1319:number_of_matches]
      df$importance1[1319:number_of_matches] <- new$importance1[1319:number_of_matches]
      df$importance2[1319:number_of_matches] <- new$importance2[1319:number_of_matches]
      df$score1[1319:number_of_matches] <- new$score1[1319:number_of_matches]
      df$score2[1319:number_of_matches] <- new$score2[1319:number_of_matches]
      df$xG1[1319:number_of_matches] <- new$xG1[1319:number_of_matches]
      df$xG2[1319:number_of_matches]<- new$xG2[1319:number_of_matches]

      df <- df %>% mutate(home_result = if_else(score1>score2, "win", if_else(score1==score2, "draw", "loss")))
      df %>% write.csv("updated.csv")
      number_of_matches <- df %>% filter(!is.na(score1)) %>% nrow()
      predictor(df, number_of_matches + 1)
      roc_curves(df, number_of_matches + 1)
      scatterer(df, number_of_matches + 1)
      controlVar$resultsReady <- TRUE
    })





    output$readytable <- renderDataTable({
        input$update
        round_df <- function(df, digits) {
            nums <- vapply(df, is.numeric, FUN.VALUE = logical(1))
            df[,nums] <- round(df[,nums], digits = digits)
            (df)
          }
        if (controlVar$updateReady){
          dat <- read_csv("updated.csv") %>% select(-X1, -X1_1, -proj_score1, -proj_score2) %>% round_df(2)
          number_of_matches <- dat %>% filter(!is.na(score1)) %>% nrow()
          datatable(dat %>% slice(0:number_of_matches),
                    filter = 'bottom', options = list(scrollX = TRUE, autoWidth = TRUE,columnDefs = list(list(targets='_all', className = 'dt-center')))) %>%
                    formatStyle('spi1', Color = styleInterval(5, c('black', '#006400')), backgroundColor = styleInterval(c(10,20,30,40,50,60,70,80,85,90,95,100), c('#e5447c', '#d65384', '#c6618d', '#b77095', '#a87f9d', '#988ea6', '#899dae', '#7aabb6', '#6ababf', '#5bc9c7', '#4cd7cf', '#3ce6d8', '#2df5e0')),fontWeight = styleInterval(5, c('normal', 'bold'))) %>%
                    formatStyle('spi2', Color = styleInterval(5, c('black', '#006400')), backgroundColor = styleInterval(c(10,20,30,40,50,60,70,80,85,90,95,100), c('#e5447c', '#d65384', '#c6618d', '#b77095', '#a87f9d', '#988ea6', '#899dae', '#7aabb6', '#6ababf', '#5bc9c7', '#4cd7cf', '#3ce6d8', '#2df5e0')),fontWeight = styleInterval(5, c('normal', 'bold'))) %>%
                    formatStyle('home_pos', Color = styleInterval(5, c('black', '#006400')), backgroundColor = styleInterval(c(1,2,3,4,7,10,12,15,17,18,19,20), c('#2df5e0', '#3ce6d8', '#4cd7cf', '#5bc9c7', '#6ababf', '#7aabb6', '#899dae', '#988ea6', '#a87f9d', '#b77095', '#c6618d', '#d65384', '#e5447c')),fontWeight = styleInterval(5, c('normal', 'bold'))) %>%
                    formatStyle('visitor_pos', Color = styleInterval(5, c('black', '#006400')), backgroundColor = styleInterval(c(1,2,3,4,7,10,12,15,17,18,19,20), c('#2df5e0', '#3ce6d8', '#4cd7cf', '#5bc9c7', '#6ababf', '#7aabb6', '#899dae', '#988ea6', '#a87f9d', '#b77095', '#c6618d', '#d65384', '#e5447c')),fontWeight = styleInterval(5, c('normal', 'bold'))) %>%
                    formatStyle('home_form', Color = styleInterval(5, c('black', '#006400')), backgroundColor = styleInterval(c(0,.4,.8,1.2,1.6,2,2.2,2.4,2.6,2.8, 2.9,3), c('#e5447c', '#d65384', '#c6618d', '#b77095', '#a87f9d', '#988ea6', '#899dae', '#7aabb6', '#6ababf', '#5bc9c7', '#4cd7cf', '#3ce6d8', '#2df5e0')),fontWeight = styleInterval(5, c('normal', 'bold'))) %>%
                    formatStyle('visitor_form', Color = styleInterval(5, c('black', '#006400')), backgroundColor = styleInterval(c(0,.4,.8,1.2,1.6,2,2.2,2.4,2.6,2.8, 2.9,3), c('#e5447c', '#d65384', '#c6618d', '#b77095', '#a87f9d', '#988ea6', '#899dae', '#7aabb6', '#6ababf', '#5bc9c7', '#4cd7cf', '#3ce6d8', '#2df5e0')),fontWeight = styleInterval(5, c('normal', 'bold'))) %>%
                    formatStyle('importance1', Color = styleInterval(2, c('black', '#006400')),fontWeight = styleInterval(2, c('normal', 'bold')),
                    background = styleColorBar(dat$importance1, '#5bc9c7'),
                    backgroundSize = '100% 90%',
                    backgroundRepeat = 'no-repeat',
                    backgroundPosition = 'center')  %>%
                    formatStyle('importance2', Color = styleInterval(2, c('black', '#006400')),fontWeight = styleInterval(2, c('normal', 'bold')),
                    background = styleColorBar(dat$importance2, '#5bc9c7'),
                    backgroundSize = '100% 90%',
                    backgroundRepeat = 'no-repeat',
                    backgroundPosition = 'center')

    }})
    output$plot1 <- renderPlotly({
      input$update
      if (controlVar$resultsReady){
        roc <- read_csv("reticulate1.csv")
        plot_ly(roc, x = ~FPR, y = ~TPR, type = 'scatter', mode = 'lines')
                 }
    })
    output$plot2 <- renderPlotly({
      input$update
      if (controlVar$resultsReady){
        pca <- read_csv("reticulate_3.csv")
        plot_ly(data = pca, x = ~x, y = ~y, color = ~label)
                 }
    })
    output$resultstable <- renderDataTable({
        input$update
        if (controlVar$resultsReady){
          pred_results <- read_csv("reticulate2.csv")
          teams_2019 <- pred_results %>% select(home) %>% unique()
          league_points = teams_2019 %>% rename("team" = "home")
          rownames(teams_2019) <- teams_2019$home

          pred_results <- pred_results %>% rename("win" = "win_prob", "draw" = "draw_prob", "loss" = "loss_prob")
          counter = 1
          for (i in teams_2019$home) {
            cursor <- pred_results %>% filter(home == i | visitor == i) %>% select(home, visitor, win, draw, loss) %>% mutate(points = if_else(home == i, win*3 + draw*1, loss*3 + draw*1)) %>% select(points) %>% sum()
            counter <- counter + 1
            league_points[i,2] = cursor
          }
          league_points <- league_points %>% select(-team)
          league_points <- league_points %>% filter(!is.na(V2))
          Predictions = tibble(team = teams_2019$home, points = league_points$V2)
          Predictions$points = round(Predictions$points,0)
          Predictions <- Predictions %>% arrange(desc(points))

          datatable(Predictions,
                         options = list(
                           paging =TRUE,
                           pageLength =  25,
                           rowCallback = JS('function(row, data, index, rowId) {',
                       '    console.log(rowId)','if(rowId >= 0 && rowId <= 3) {',
                            'row.style.backgroundColor = "PaleGreen";',
                            '} ','else if (rowId >= 17 && rowId <= 19) {',
                                 'row.style.backgroundColor = "Pink";',
                                 '}',
                                 '}')
    )
                           )

        }
    })



}
)

shinyApp(ui = ui, server = server)
