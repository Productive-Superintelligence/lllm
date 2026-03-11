import os
from lllm import Orchestrator, Prompt
from lllm.proxies.base import PROXY_REGISTRY

# Import the builtin proxy module. 
# The @ProxyRegistrator decorator in the module will automatically register the proxy 
# into PROXY_REGISTRY when the module is imported.
from lllm.proxies.builtin import exa_proxy

def main():
    # Verify registration
    print("Registered Proxies:", list(PROXY_REGISTRY.keys()))
    
    if 'exa' not in PROXY_REGISTRY:
        print("Error: Exa proxy not registered.")
        return

    # In a real scenario, you would configure the agent to use this proxy.
    # The proxy is available in the registry and can be instantiated by the Agent or Sandbox.
    
    print("Exa Proxy class:", PROXY_REGISTRY['exa'])
    
    # Example of manual instantiation (usually handled by the framework)
    # exa = PROXY_REGISTRY['exa'](cutoff_date=None)
    # print("Exa Proxy instance created.")

if __name__ == "__main__":
    main()
