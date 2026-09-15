function(add_example name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE MLlib)
endfunction()

add_example(markowitz)
add_example(markowitz_ad)