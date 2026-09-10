add_executable(test_simple_ad test_simple_ad.cpp)
target_link_libraries(test_simple_ad PRIVATE MLlib)

add_test(NAME test_simple_ad COMMAND test_simple_ad)