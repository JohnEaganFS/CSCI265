import pstats

# Load the stats file (output.pstats)
p = pstats.Stats('./profiling_output/output.pstats')

# Sort the stats by the cumulative time spent in the function
p.sort_stats('cumulative')

# Print the top 10 functions
p.print_stats(40)