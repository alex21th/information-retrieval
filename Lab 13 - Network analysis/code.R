library(igraph)


edges <- scan('edges.txt')

edges <- matrix(edges+1, ncol = 2, byrow = TRUE)

graph <- graph_from_edgelist(edges, directed = FALSE)


# edge betweenness
eb <- edge.betweenness.community(graph)

# vertices
gorder(graph)
# edges
gsize(graph)
# diameter
diameter(graph)
# transitivity
transitivity(graph)
# degree distribution
a <- degree.distribution(graph)
length(a[a > 0])

modularity(eb) # 0.4478

# pagerank
pr <- page_rank(graph)
plot(graph, vertex.size = pr$vector*400)

###
# COMMUNITIES
###
library(ggplot2)

# Walktrap [Pons and Latapy, 2005]
walktrap <- walktrap.community(graph)

# Edge betweenness [Newman and Girvan, 2004]
betweenness <- edge.betweenness.community(graph)

# Multilevel [Blondel et al., 2008]
multilevel <- multilevel.community(graph)

# Optimal [Brandes et al., 2008]
optimal <- optimal.community(graph)

plot(walktrap, graph, main='Walktrap')
plot(betweenness, graph, main='Edge betweenness')
plot(multilevel, graph, main='Multilevel')
plot(optimal, graph, main='Optimal')

ggplot() + aes(membership(walktrap))+ geom_histogram(binwidth=1, colour="white", fill="orange")
ggplot() + aes(membership(betweenness))+ geom_histogram(binwidth=1, colour="white", fill="blue")
ggplot() + aes(membership(multilevel))+ geom_histogram(binwidth=1, colour="white", fill="green")
ggplot() + aes(membership(optimal))+ geom_histogram(binwidth=1, colour="white", fill="purple")
