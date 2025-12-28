"""
Inspect streamlit-authenticator installation and Authenticate.login signature.
Run inside your app venv: python auth_debug.py
"""
import inspect
import streamlit_authenticator as stauth
import importlib

print('module:', stauth.__name__)
print('file:', getattr(stauth, '__file__', 'N/A'))
print('version attr:', getattr(stauth, '__version__', 'N/A'))

# Try to import version from pkg_resources if available
try:
    import pkg_resources
    ver = pkg_resources.get_distribution('streamlit-authenticator').version
    print('pkg_resources version:', ver)
except Exception:
    pass

# Inspect Authenticate
if hasattr(stauth, 'Authenticate'):
    Auth = stauth.Authenticate
    print('\nAuthenticate object:', Auth)
    try:
        sig = inspect.signature(Auth)
        print('Authenticate signature:', sig)
    except Exception as e:
        print('Could not get signature for Authenticate:', e)

    # Inspect login method
    try:
        login_fn = getattr(Auth, 'login', None)
        print('\nAuthenticate.login attribute on class:', login_fn)
        if login_fn is not None:
            try:
                print('login signature (callable):', inspect.signature(login_fn))
            except Exception as e:
                print('Could not get login signature:', e)
            try:
                print('\nlogin docstring:\n', inspect.getdoc(login_fn))
            except Exception:
                pass
    except Exception as e:
        print('Error inspecting login:', e)
else:
    print('streamlit_authenticator has no Authenticate attribute')

print('\nDone')
