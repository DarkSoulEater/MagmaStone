#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <optional>
#include <set>
#include <fstream>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;


const std::vector<const char*> validation_layers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> device_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
  const bool enable_validations_layers = false;
#else
  const bool enable_validations_layers = true;
#endif


VkResult CreateDebugUtilsMessengerEXT(
                        VkInstance instance,
                        const VkDebugUtilsMessengerCreateInfoEXT* p_create_info,
                        const VkAllocationCallbacks* p_allocator,
                        VkDebugUtilsMessengerEXT* p_debug_messenger) {

  auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, p_create_info, p_allocator, p_debug_messenger);
  } else {
    throw VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(
            VkInstance instance,
            VkDebugUtilsMessengerEXT debug_messenger,
            const VkAllocationCallbacks* p_allocator) {

  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debug_messenger, p_allocator);
  }
}


struct QueueFamilyIndices {
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> present_family;

  bool isComplete() {
    return graphics_family.has_value() && present_family.has_value();
  }
};

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

class HelloTriangleApplication {
 public:
  void Run() {
    InitWindow();
    InitVulkan();
    MainLoop();
    Cleanup();
  }

 private:
  GLFWwindow *window_;

  VkInstance instance_;
  VkDebugUtilsMessengerEXT debug_messenger_;
  VkSurfaceKHR surface_;

  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_;

  VkQueue graphics_queue_;
  VkQueue present_queue_;

  VkSwapchainKHR swap_chain_;
  std::vector<VkImage> swap_chain_images_;
  VkFormat swap_chain_image_format_;
  VkExtent2D swap_chain_extent_;
  std::vector<VkImageView> swap_chain_image_views_;

  VkRenderPass render_pass_;
  VkPipelineLayout pipeline_layout_;
  VkPipeline graphics_pipeline_;

  void InitWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window_ = glfwCreateWindow(WIDTH, HEIGHT, "MagmaStone", nullptr, nullptr);
  }

  void InitVulkan() {
    CreateInstance();
    SetupDebugMessenger();
    CreateSurface();
    PickPhysicalDevice();
    CreateLogicalDevice();
    CreateSwapChain();
    CreateImageViews();
    CreateRenderPass();
    CreateGraphicsPipeline();
  }

  void MainLoop() {
    while (!glfwWindowShouldClose(window_)) {
      glfwPollEvents();
    }
  }

  void Cleanup() {
    vkDestroyPipeline(device_, graphics_pipeline_, nullptr);
    vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);

    for (auto image_view : swap_chain_image_views_) {
      vkDestroyImageView(device_, image_view, nullptr);
    }

    vkDestroySwapchainKHR(device_, swap_chain_, nullptr);
    vkDestroyDevice(device_, nullptr);

    if (enable_validations_layers) {
      DestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
    }

    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    vkDestroyInstance(instance_, nullptr);

    glfwDestroyWindow(window_);

    glfwTerminate();
  }


  void CreateInstance() {
    if (enable_validations_layers && !CheckValidationLayerSupport()) {
      throw std::runtime_error("validation layers requested, but not available!");
    }

    VkApplicationInfo app_info;
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Hellow triangle";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    auto extensions = GetRequiredExtensions();
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debug_create_info;
    if (enable_validations_layers) {
     create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
     create_info.ppEnabledLayerNames = validation_layers.data();

     PopulateDebugMessengerCreateInfo(debug_create_info);
     create_info.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debug_create_info;
    } else {
      create_info.enabledLayerCount = 0;

      create_info.pNext = nullptr;
    }

    if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create instance!");
    }
    /*
     // Выводит поддерживаемые расширения
    uint32_t extensionCount = 0;

    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> extensions_(extensionCount);

    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions_.data());
    std::cout << "available extensions:\n";

    for (const auto& extension : extensions_) {
      std::cout << '\t' << extension.extensionName << '\n';
    }
  */
  }

  void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& create_info) {
    create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = DebugCallback;
    create_info.pUserData = nullptr; // optional
  }

  void SetupDebugMessenger() {
    if (!enable_validations_layers) return;

    VkDebugUtilsMessengerCreateInfoEXT create_info;
    PopulateDebugMessengerCreateInfo(create_info);

    if (CreateDebugUtilsMessengerEXT(instance_, &create_info, nullptr, &debug_messenger_) != VK_SUCCESS) {
      throw std::runtime_error("failed to set up debug messenger!");
    }
  }


  void CreateSurface() {
    if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
  }

  void PickPhysicalDevice() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);

    if (device_count == 0) {
      throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    for (const auto& device : devices) {
      if (IsDeviceSuitable(device)) {
        physical_device_ = device;
        break;
      }
    }

    if (physical_device_ == VK_NULL_HANDLE) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
  }

  void CreateLogicalDevice() {
    QueueFamilyIndices indices = FindQueueFamilies(physical_device_);

    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    std::set<uint32_t> unique_queue_families = {
        indices.graphics_family.value(),
        indices.present_family.value()
    };

    float queue_priority = 1.0f;
    for (uint32_t queue_family : unique_queue_families) {
      VkDeviceQueueCreateInfo queue_create_info{};
      queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queue_create_info.queueFamilyIndex = queue_family;
      queue_create_info.queueCount = 1;
      queue_create_info.pQueuePriorities = &queue_priority;
      queue_create_infos.push_back(queue_create_info);
    }

    VkPhysicalDeviceFeatures device_features{};

    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
    create_info.pQueueCreateInfos = queue_create_infos.data();

    create_info.pEnabledFeatures = &device_features;

    create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    create_info.ppEnabledExtensionNames = device_extensions.data();

    if (enable_validations_layers) {
      create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
      create_info.ppEnabledLayerNames = validation_layers.data();
    } else {
      create_info.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device_, indices.graphics_family.value(), 0, &graphics_queue_);
    vkGetDeviceQueue(device_, indices.present_family.value(), 0, &present_queue_);
  }

  void CreateSwapChain() {
    SwapChainSupportDetails swap_chain_support = QuerySwapChainSupport(physical_device_);

    VkSurfaceFormatKHR surface_format = ChooseSwapSurfaceFormat(swap_chain_support.formats);
    VkPresentModeKHR present_mode = ChooseSwapPresentMode(swap_chain_support.present_modes);
    VkExtent2D extent = ChooseSwapExtent(swap_chain_support.capabilities);

    uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
    if (    swap_chain_support.capabilities.maxImageCount > 0
         && image_count > swap_chain_support.capabilities.maxImageCount) {
      image_count = swap_chain_support.capabilities.minImageCount;
    }

    VkSwapchainCreateInfoKHR create_info {};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = surface_;

    create_info.minImageCount = image_count;
    create_info.imageFormat = surface_format.format;
    create_info.imageColorSpace = surface_format.colorSpace;
    create_info.imageExtent = extent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = FindQueueFamilies(physical_device_);
    uint32_t queue_family_indices[] = {indices.graphics_family.value(), indices.present_family.value()};

    if (indices.graphics_family != indices.present_family) {
      create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      create_info.queueFamilyIndexCount = 2;
      create_info.pQueueFamilyIndices = queue_family_indices;
    } else {
      create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
      create_info.queueFamilyIndexCount = 0;
      create_info.pQueueFamilyIndices = nullptr;
    }

    create_info.preTransform = swap_chain_support.capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE;

    create_info.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swap_chain_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device_, swap_chain_, &image_count, nullptr);
    swap_chain_images_.resize(image_count);
    vkGetSwapchainImagesKHR(device_, swap_chain_, &image_count, swap_chain_images_.data());

    swap_chain_image_format_ = surface_format.format;
    swap_chain_extent_= extent;
  }

  void CreateImageViews() {
    swap_chain_image_views_.resize(swap_chain_images_.size());

    for (size_t i = 0; i < swap_chain_images_.size(); ++i) {
      VkImageViewCreateInfo create_info{};
      create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      create_info.image = swap_chain_images_[i];

      create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      create_info.format = swap_chain_image_format_;

      create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
      create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
      create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
      create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

      create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      create_info.subresourceRange.baseMipLevel = 0;
      create_info.subresourceRange.levelCount = 1;
      create_info.subresourceRange.baseArrayLayer = 0;
      create_info.subresourceRange.layerCount = 1;

      if (vkCreateImageView(device_, &create_info, nullptr, &swap_chain_image_views_[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image views!");
      }
    }
  }

  void CreateRenderPass() {
    VkAttachmentDescription color_attachment{};
    color_attachment.format = swap_chain_image_format_;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_ref{};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;

    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = 1;
    render_pass_info.pAttachments = &color_attachment;
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;

    if (vkCreateRenderPass(device_, &render_pass_info, nullptr, &render_pass_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create render pass!");
    }
  }

  void CreateGraphicsPipeline() {
    auto vert_shader_code = ReadFile("../shaders/vert.spv");
    auto frag_shader_code = ReadFile("../shaders/frag.spv");

    VkShaderModule vert_shader_module = CreateShaderModule(vert_shader_code);
    VkShaderModule frag_shader_module = CreateShaderModule(frag_shader_code);


    VkPipelineShaderStageCreateInfo vert_shader_stage_info{};
    vert_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_shader_stage_info.module = vert_shader_module;
    vert_shader_stage_info.pName = "main";

    VkPipelineShaderStageCreateInfo frag_shader_stage_info{};
    frag_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_shader_stage_info.module = frag_shader_module;
    frag_shader_stage_info.pName = "main";

    VkPipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info, frag_shader_stage_info};

    // The structure describes the format of the vertex data that will be passed to the vertex shader.
    VkPipelineVertexInputStateCreateInfo vertex_input_info{};
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info.vertexBindingDescriptionCount = 0;
    vertex_input_info.pVertexBindingDescriptions = nullptr; // Optional
    vertex_input_info.vertexAttributeDescriptionCount = 0;
    vertex_input_info.pVertexAttributeDescriptions = nullptr; // Optional

    // The struct describes two things: what kind of geometry will be drawn from the vertices and if primitive restart should be enabled.
    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;


    // Viewport
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) swap_chain_extent_.width;
    viewport.height = (float) swap_chain_extent_.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    // Rect2D
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swap_chain_extent_;

    // United viewport and scissor
    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;


    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional


    // Multisampling
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // Optional


    // Color blending
    VkPipelineColorBlendAttachmentState color_blend_attachment{};
    color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment.blendEnable = VK_FALSE;
    //color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    //color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    //color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
    //color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    //color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    //color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

    VkPipelineColorBlendStateCreateInfo color_blending{};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = VK_LOGIC_OP_COPY; // Optional
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &color_blend_attachment;
    color_blending.blendConstants[0] = 0.0f; // Optional
    color_blending.blendConstants[1] = 0.0f; // Optional
    color_blending.blendConstants[2] = 0.0f; // Optional
    color_blending.blendConstants[3] = 0.0f; // Optional

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 0; // Optional
    pipeline_layout_info.pSetLayouts = nullptr; // Optional
    pipeline_layout_info.pushConstantRangeCount = 0; // Optional
    pipeline_layout_info.pPushConstantRanges = nullptr; // Optional

    if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = shader_stages;

    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pDepthStencilState = nullptr;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = nullptr;

    pipeline_info.layout = pipeline_layout_;
    pipeline_info.renderPass = render_pass_;
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_info.basePipelineIndex = -1;

    if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline_) != VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device_, frag_shader_module, nullptr);
    vkDestroyShaderModule(device_, vert_shader_module, nullptr);
  }

  VkShaderModule CreateShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code.size();
    create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shader_module;
    if (vkCreateShaderModule(device_, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
      throw std::runtime_error("failed to create shader module!");
    }

    return shader_module;
  }

  SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;

    // Query capabilities
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);

    // Query format
    uint32_t format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count, nullptr);

    if (format_count != 0) {
      details.formats.resize(format_count);
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count, details.formats.data());
    }

    // Query present mode
    uint32_t present_mode_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &present_mode_count, nullptr);

    if (present_mode_count != 0) {
      details.present_modes.resize(present_mode_count);
      vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &present_mode_count, details.present_modes.data());
    }

    return details;
  }

  bool IsDeviceSuitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = FindQueueFamilies(device);

    bool extensions_supported = CheckDeviceExtensionsSupport(device);

    bool swap_chain_adequate = false;
    if (extensions_supported) {
      SwapChainSupportDetails swap_chain_support = QuerySwapChainSupport(device);
      swap_chain_adequate = !swap_chain_support.formats.empty() && !swap_chain_support.present_modes.empty();
    }

    return indices.isComplete() && extensions_supported && swap_chain_adequate;
  }

  bool CheckDeviceExtensionsSupport(VkPhysicalDevice device) {
    uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());

    std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());
    for (const auto& extension : available_extensions) {
      required_extensions.erase(extension.extensionName);
    }

    return required_extensions.empty();
  }


  VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &available_formats) {
    for (const auto &available_format : available_formats) {
      if (    available_format.format == VK_FORMAT_B8G8R8_SRGB
           && available_format.colorSpace == VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT) {
        return available_format;
      }
    }

    return available_formats[0];
  }

  VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR> &available_present_modes) {
    for (const auto& available_present_mode : available_present_modes) {
      if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
        return available_present_mode;
      }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
  }

  VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX){
      return capabilities.currentExtent;
    } else {
      int width, height;
      glfwGetFramebufferSize(window_, &width, &height);

      VkExtent2D actual_extent = {
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height)
      };

      actual_extent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actual_extent.width));
      actual_extent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actual_extent.height));

      return actual_extent;
    }
  }


  QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

    // Get queue family
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

    int i = 0;
    for (const auto& queue_family : queue_families) {
      if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        indices.graphics_family = i;
      }

      VkBool32 present_support = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &present_support);
      if (present_support) {
        indices.present_family = i;
      }

      if (indices.isComplete()) {
        break;
      }
      ++i;
    }

    return indices;
  }


  std::vector<const char*> GetRequiredExtensions() {
    uint32_t glfw_extension_count = 0;
    const char** glfw_extensions;
    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

    if (enable_validations_layers) {
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
  }

  bool CheckValidationLayerSupport() {
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    for (const char* layer_name : validation_layers) {
      bool layer_found = false;

      for (const auto &layer_properties: available_layers) {
        if (strcmp(layer_name, layer_properties.layerName) == 0) {
          layer_found = true;
          break;
        }
      }

      if (!layer_found) {
        return false;
      }
    }

    return true;
  }

  static std::vector<char> ReadFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      throw std::runtime_error("failed to open file!");
    }

    size_t file_size = (size_t) file.tellg();
    std::vector<char> buffer(file_size);

    file.seekg(0);
    file.read(buffer.data(), file_size);

    file.close();

    return buffer;
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
      VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
      VkDebugUtilsMessageTypeFlagsEXT message_type,
      const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
      void* p_user_data) {

    std::cerr << "validation layer: " << p_callback_data->pMessage << std::endl;

    return VK_FALSE;
  }
};

int main() {
  HelloTriangleApplication app;
  try {
    app.Run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}