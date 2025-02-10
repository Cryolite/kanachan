add_library(_nyanten SHARED)
target_sources(_nyanten
  PRIVATE
  nyanten.cpp
)
target_link_libraries(_nyanten
  PRIVATE Boost::python
  PRIVATE Boost::headers
  PRIVATE "python${PYTHON_VERSION}")
