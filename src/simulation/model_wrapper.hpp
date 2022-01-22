#if !defined(KANACHAN_SIMULATION_MODEL_WRAPPER_HPP_INCLUDE_GUARD)
#define KANACHAN_SIMULATION_MODEL_WRAPPER_HPP_INCLUDE_GUARD

#include <boost/python/object.hpp>
#include <string>
#include <cstdint>


namespace Kanachan{

class ModelWrapper;

void swap(ModelWrapper &lhs, ModelWrapper &rhs) noexcept;

class ModelWrapper
{
public:
  ModelWrapper(
    std::string const &device, boost::python::object dtype,
    boost::python::object model);

  ModelWrapper(ModelWrapper const &rhs) = default;

  ModelWrapper(ModelWrapper &&rhs) noexcept = default;

  void swap(ModelWrapper &rhs) noexcept;

  ModelWrapper &operator=(ModelWrapper const &rhs);

  ModelWrapper &operator=(ModelWrapper &&rhs) noexcept;

public:
  std::uint_fast16_t operator()(boost::python::object features) const;

private:
  std::string device_;
  boost::python::object dtype_;
  boost::python::object model_;
}; // class ModelWrapper

} // namespace Kanachan


#endif // KANACHAN_SIMULATION_MODEL_WRAPPER_HPP_INCLUDE_GUARD
