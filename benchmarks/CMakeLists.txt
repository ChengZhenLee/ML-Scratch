find_package(benchmark REQUIRED)

function(add_benchmark name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE benchmark::benchmark pthread MLlib)
endfunction()

add_benchmark(ad_vs_cfd)