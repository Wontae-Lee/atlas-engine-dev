#pragma once

#include <atlas/system/system.h>

#include <functional>

namespace atlas::interactive {

using SystemFactory = std::function<atlas::SystemHostPtr()>;

}
