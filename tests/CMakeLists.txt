function(add_ml_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE MLlib)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_ml_test(test_simple_ad)
add_ml_test(test_black_scholes)