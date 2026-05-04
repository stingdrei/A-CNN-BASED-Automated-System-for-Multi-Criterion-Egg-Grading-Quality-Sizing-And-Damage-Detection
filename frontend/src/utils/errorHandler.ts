import axios, { AxiosError } from 'axios';

/**
 * Extracts a user-friendly error message from an unknown error.
 * Prioritizes FastAPI's error response format (error.response.data.detail).
 * 
 * @param error - The caught error (typically from axios)
 * @returns A string error message safe to display to the user
 */
export function getErrorMessage(error: unknown): string {
  // Handle Axios errors (API requests)
  if (axios.isAxiosError(error)) {
    const responseData = error.response?.data;
    
    // FastAPI typically returns errors in `detail` field
    if (responseData?.detail) {
      // Handle array of validation errors (FastAPI returns array for validation errors)
      if (Array.isArray(responseData.detail)) {
        return responseData.detail
          .map((d: any) => d.msg || d.message || JSON.stringify(d))
          .join(', ');
      }
      // Handle string detail (typical error)
      if (typeof responseData.detail === 'string') {
        return responseData.detail;
      }
      // Handle object detail
      if (typeof responseData.detail === 'object') {
        return JSON.stringify(responseData.detail);
      }
    }
    
    // Check for message field
    if (responseData?.message) {
      return responseData.message;
    }
    
    // Fall back to HTTP status text
    if (error.response?.statusText) {
      return error.response.statusText;
    }
    
    // Fall back to axios error message
    if (error.message) {
      return error.message;
    }
    
    return 'An error occurred with the request';
  }
  
  // Handle standard Error objects
  if (error instanceof Error) {
    return error.message;
  }
  
  // Handle string errors
  if (typeof error === 'string') {
    return error;
  }
  
  // Last resort
  return 'An unexpected error occurred';
}
